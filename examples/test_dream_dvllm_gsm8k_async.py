import asyncio
import time
from typing import List

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from d2f_engine.engine.async_engine import AsyncEngine
from d2f_engine.sampling_params import SamplingParams


FEW_SHOTS = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

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
    # Remove leading spaces, exclamation marks, and newlines
    return text.lstrip(" !\n")


async def process_sequential(
    engine: AsyncEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer
) -> List[str]:
    """Process prompts sequentially"""
    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Processing sequentially")):
        seq_id = engine.add_request(prompt, sampling_params)
        result = await collect_streaming_result(engine, seq_id, tokenizer)
        results.append(result)
    return results


async def process_concurrent(
    engine: AsyncEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer,
    batch_size: int = None
) -> List[str]:
    """Process prompts concurrently (optionally in batches)"""
    if batch_size is None:
        # Process all at once
        seq_ids = []
        for prompt in tqdm(prompts, desc="Adding requests"):
            seq_id = engine.add_request(prompt, sampling_params)
            seq_ids.append(seq_id)
        
        tasks = [collect_streaming_result(engine, seq_id, tokenizer) for seq_id in seq_ids]
        results = await asyncio.gather(*tasks)
        return results
    else:
        # Process in batches
        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
            batch_prompts = prompts[i:i + batch_size]
            seq_ids = []
            for prompt in batch_prompts:
                seq_id = engine.add_request(prompt, sampling_params)
                seq_ids.append(seq_id)
            
            tasks = [collect_streaming_result(engine, seq_id, tokenizer) for seq_id in seq_ids]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results


def count_tokens(texts: List[str], tokenizer: AutoTokenizer) -> int:
    """Count total tokens in texts"""
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)


async def main():
    model = "/home/ljp/models/Dream-v0-Base-7B/"
    lora_path = "/home/ljp/models/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora/"
    
    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("/home/ljp/datasets/gsm8k", "main")['test']
    questions = dataset['question']
    # Limit to first N questions for testing (adjust as needed)
    num_questions = len(questions)  # Use all questions, or set a limit like [:10]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    bos = tokenizer.bos_token or ""
    
    # Prepare prompts
    prompts = [bos + FEW_SHOTS + q for q in questions[:num_questions]]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    # Determine max_num_seqs based on whether we'll process concurrently
    max_num_seqs = max(4, min(20, len(prompts)))  # Cap at 20 for memory efficiency
    
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
        max_num_seqs=max_num_seqs,
        max_model_len=1024,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
    ) as engine:
        # Warmup
        print("Warming up with a dummy request...")
        dummy_prompt = bos + "Q: 1 + 1 = ?\nA:"
        dummy_id = engine.add_request(dummy_prompt, sampling_params)
        _ = await collect_streaming_result(engine, dummy_id, tokenizer)
        print("Warmup complete.\n")
        
        # Choose processing mode
        use_concurrent = True  # Set to False for sequential processing
        concurrent_batch_size = max_num_seqs  # Process in batches if needed
        
        if use_concurrent:
            print("="*50)
            print("CONCURRENT PROCESSING")
            print("="*50)
            start_time = time.time()
            results = await process_concurrent(
                engine, prompts, sampling_params, tokenizer, 
                batch_size=concurrent_batch_size if len(prompts) > concurrent_batch_size else None
            )
            total_time = time.time() - start_time
        else:
            print("="*50)
            print("SEQUENTIAL PROCESSING")
            print("="*50)
            start_time = time.time()
            results = await process_sequential(engine, prompts, sampling_params, tokenizer)
            total_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens = count_tokens(results, tokenizer)
        avg_tokens_per_output = total_tokens / len(results) if results else 0
        
        # Print results
        print("\n" + "=*=" * 30)
        print("PROFILING RESULTS")
        print("=*=" * 30)
        print(f"Generated {len(results)} outputs.")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per output: {avg_tokens_per_output:.2f}")
        print(f"Total time: {total_time:.2f} seconds.")
        print(f"Average TPS: {total_tokens / total_time:.2f} tok/s.")
        print(f"Average latency per request: {total_time / len(results):.2f} seconds.")
        print("=*=" * 30)
        
        # Print sample results
        print("\n" + "=*=" * 30)
        print("SAMPLE RESULTS")
        print("=*=" * 30)
        for idx in range(min(3, len(results))):
            print(f"\n[Prompt {idx} Result]")
            print(f"Question: {questions[idx]}")
            print(f"Answer: {results[idx][:500]}{'...' if len(results[idx]) > 500 else ''}")
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())


