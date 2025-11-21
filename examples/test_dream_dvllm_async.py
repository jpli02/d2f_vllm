import asyncio
import time
from typing import List

from transformers import AutoTokenizer
from d2f_engine.engine.async_engine import AsyncEngine
from d2f_engine.sampling_params import SamplingParams


FEW_SHOTS = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
"""


async def collect_streaming_result(engine: AsyncEngine, seq_id: int, tokenizer: AutoTokenizer) -> str:
    """Collect streaming tokens and return decoded text"""
    tokens = []
    async for toks, finished in engine.stream(seq_id):
        if toks:
            tokens.extend(toks)
        if finished:
            break

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    # remove leading ! and newlines
    return text.lstrip(" !\n")


async def test_sequential_streaming(
    engine: AsyncEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer
) -> List[str]:
    """Test streaming one sequence at a time"""
    print("Testing sequential streaming...")
    results = []

    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        seq_id = engine.add_request(prompt, sampling_params)
        result = await collect_streaming_result(engine, seq_id, tokenizer)
        results.append(result)

    return results


async def test_concurrent_streaming(
    engine: AsyncEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer
) -> List[str]:
    """Test streaming all sequences concurrently"""
    print("Testing concurrent streaming...")

    # Add all requests
    seq_ids = []
    for prompt in prompts:
        seq_id = engine.add_request(prompt, sampling_params)
        seq_ids.append(seq_id)

    # Process all concurrently
    tasks = [collect_streaming_result(engine, seq_id, tokenizer) for seq_id in seq_ids]
    results = await asyncio.gather(*tasks)

    return results


def count_tokens(texts: List[str], tokenizer: AutoTokenizer) -> int:
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)


async def main():
    model = "/home/ljp/models/Dream-v0-Base-7B/"
    lora_path = "/home/ljp/models/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora/"

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    bos = tokenizer.bos_token or ""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    # Test prompts
    test_prompts = [
        bos + FEW_SHOTS + "Q: What is 2 + 2?\nA:",
        bos + FEW_SHOTS + "Q: If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?\nA:",
        bos + FEW_SHOTS + "Q: A store has 100 apples. They sell 30 apples. How many apples are left?\nA:",
    ]

    async with AsyncEngine(
        model,
        lora_path=lora_path,
        use_lora=True,
        model_name="dream",
        model_type="diffusion_lm",
        enforce_eager=True,
        data_parallel_size=4,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_num_batched_tokens=1024,
        max_num_seqs=max(4, len(test_prompts)),
        max_model_len=1024,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
    ) as engine:
        # real warmup: run a short request, not just sleep
        print("Warming up with a dummy request...")
        dummy_prompt = bos + "Q: 1 + 1 = ?\nA:"
        dummy_id = engine.add_request(dummy_prompt, sampling_params)
        _ = await collect_streaming_result(engine, dummy_id, tokenizer)

        # Test sequential
        print("\n" + "="*50)
        print("SEQUENTIAL STREAMING TEST")
        print("="*50)
        start_time = time.time()
        seq_results = await test_sequential_streaming(engine, test_prompts, sampling_params, tokenizer)
        seq_time = time.time() - start_time

        print(f"Sequential time: {seq_time:.2f}s")
        for i, result in enumerate(seq_results):
            print(f"\nResult {i+1}:")
            print(result[:200] + "..." if len(result) > 200 else result)

        # Test concurrent
        print("\n" + "="*50)
        print("CONCURRENT STREAMING TEST")
        print("="*50)
        start_time = time.time()
        concurrent_results = await test_concurrent_streaming(engine, test_prompts, sampling_params, tokenizer)
        concurrent_time = time.time() - start_time

        print(f"Concurrent time: {concurrent_time:.2f}s")
        for i, result in enumerate(concurrent_results):
            print(f"\nResult {i+1}:")
            print(result[:200] + "..." if len(result) > 200 else result)

        # Performance comparison
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        if concurrent_time > 0:
            speedup = seq_time / concurrent_time
            print(f"Sequential time: {seq_time:.2f}s")
            print(f"Concurrent time: {concurrent_time:.2f}s")
            print(f"Speedup (wall-clock): {speedup:.2f}x")

            # summary
            seq_tokens = count_tokens(seq_results, tokenizer)
            conc_tokens = count_tokens(concurrent_results, tokenizer)
            print(f"Sequential avg latency per request: {seq_time / len(test_prompts):.2f}s")
            print(f"Concurrent avg latency per request: {concurrent_time / len(test_prompts):.2f}s")
            print(f"Sequential throughput: {seq_tokens / seq_time:.2f} tok/s")
            print(f"Concurrent throughput: {conc_tokens / concurrent_time:.2f} tok/s")
        else:
            print("Could not calculate speedup (concurrent_time is 0?)")


if __name__ == "__main__":
    asyncio.run(main())
