import atexit

import torch.multiprocessing as mp

from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import List

from d2f_engine.config import Config
from d2f_engine.sampling_params import SamplingParams
from d2f_engine.engine.sequence import SequenceForCausalLM, SequenceForDiffusionLM
from d2f_engine.engine.scheduler import AutoScheduler, SchedulerBase
from d2f_engine.engine.model_runner import AutoModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        self.engine_type = config.model_type
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=AutoModelRunner.from_config, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = AutoModelRunner.from_config(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler: SchedulerBase = AutoScheduler.from_config(config)
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, "_exited", False):
            return
        self._exited = True
        if hasattr(self, "model_runner") and self.model_runner is not None:
            try:
                self.model_runner.call("exit")
            except Exception:
                pass
            try:
                del self.model_runner
            except Exception:
                pass
        for p in getattr(self, "ps", []):
            try:
                p.join()
            except Exception:
                pass

    def add_request(self, prompt: str | List[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
            
        if self.engine_type == "causal_lm":
            seq = SequenceForCausalLM(prompt, sampling_params)
        elif self.engine_type == "diffusion_lm":
            seq = SequenceForDiffusionLM(prompt, sampling_params, config=self.config)
        else:
            raise ValueError(f"Unsupported engine type: {self.engine_type}")
        
        seq.block_size = self.config.kvcache_block_size
        self.scheduler.add(seq)
        # Return seq_id so caller can build a stable mapping
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        sample_output = self.model_runner.call("run", seqs, is_prefill)
        n_diff_steps = self.scheduler.postprocess(seqs, sample_output)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if self.engine_type == "causal_lm":
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else len(seqs)
            # For streaming: provide per-seq deltas (newly appended token) on decode steps
            if not is_prefill:
                deltas = [(seq.seq_id, [seq.last_token], seq.is_finished) for seq in seqs]
            else:
                deltas = []
        else:
            num_tokens = sum(seq.input_num_tokens + seq.new_tokens for seq in seqs) if is_prefill else sum(seq.new_tokens for seq in seqs)
            # Diffusion decoding modifies tokens in-place; we currently don't stream intermediate edits
            deltas = []
        return outputs, num_tokens, is_prefill, n_diff_steps, deltas

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
        use_tqdm: bool = True,
    ) -> List[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # Map internal seq_id -> input index to keep output order stable
        seqid_to_idx = {}
        for idx, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            sid = self.add_request(prompt, sp)
            seqid_to_idx[sid] = idx
        outputs = [None] * len(prompts)
        prefill_throughput = decode_throughput = 0.
        n_steps = 0
        n_diff_steps = [-1] * len(prompts)
        while not self.is_finished():
            t = perf_counter()
            n_steps += 1
            output, num_tokens, is_prefill, cur_n_diff_steps, _ = self.step()
            if use_tqdm:
                if is_prefill:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            if cur_n_diff_steps:
                for seq_id, n_step in cur_n_diff_steps.items():
                    if seq_id in seqid_to_idx and n_step >= 0:
                        n_diff_steps[seqid_to_idx[seq_id]] = n_step
            for seq_id, token_ids in output:
                if seq_id in seqid_to_idx:
                    outputs[seqid_to_idx[seq_id]] = token_ids
                if use_tqdm:
                    pbar.update(1)
        print(f"Finished in {n_steps} steps, prefill throughput: {prefill_throughput:.2f} tok/s, decode throughput: {decode_throughput:.2f} tok/s")
        # Ensure all outputs are present
        assert all(toks is not None for toks in outputs), "Some sequences did not produce outputs"
        outputs = [{
            "text": self.tokenizer.decode(token_ids).split(self.tokenizer.eos_token)[0],
            "token_ids": token_ids[:token_ids.index(self.config.eos)] if self.config.eos in token_ids else token_ids,
            "n_diff_steps": n_diff_step,
        } for token_ids, n_diff_step in zip(outputs, n_diff_steps)]
        if use_tqdm:
            pbar.close()
        return outputs
