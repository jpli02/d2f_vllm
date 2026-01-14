# PR Summary: Async Inference Feature Implementation

## Overview
This PR implements and improves async inference capabilities for Diffulex, enabling non-blocking concurrent request handling while maintaining thread safety and correctness.

## Changes Summary

### 1. Async Inference Implementation (`diffulex/engine/tp_worker.py`)

#### Enhanced `generate_async()` method
- **Improved async consistency**: Changed to use `is_finished_async()` in the while loop condition for full async operation
- **Thread-safe request addition**: Requests are added synchronously to avoid race conditions with the scheduler, while inference steps remain async
- **Proper async control flow**: Added `await asyncio.sleep(0)` to yield control to other async tasks

#### Enhanced `step_async()` method
- **Thread safety**: Runs the entire step (scheduler.schedule(), model_runner.call(), scheduler.postprocess()) in a single-threaded executor to prevent race conditions
- **Atomic operations**: Ensures scheduler operations are atomic and sequence state remains consistent
- **Executor management**: Uses a dedicated ThreadPoolExecutor with max_workers=1 for thread-safe execution

### 2. Bug Fixes in Sampler (`diffulex/sampler/base.py`)

#### Fixed `_fetch_last_logits()` method
- **KeyError fix**: Handles cases where `has_to_cache_block` is False and the key doesn't exist in the map
- **String key consistency**: Uses string keys (`str(seq.seq_id)`) to match the type hint `dict[str, torch.Tensor]` and usage in `fast_dllm_v2.py`
- **Fallback logic**: Provides fallback to use the last logit from the current batch when no cached value exists
- **Error handling**: Raises a clear error message when logits are empty

**Before:**
```python
def _fetch_last_logits(self, logits: torch.Tensor, seq: SequenceBase) -> torch.Tensor:
    if seq.has_to_cache_block:
        last_logits = logits[seq.to_cache_last_token_id]
        self.seq_last_logits_map[seq.seq_id] = last_logits
    return self.seq_last_logits_map[seq.seq_id]  # KeyError if key doesn't exist
```

**After:**
```python
def _fetch_last_logits(self, logits: torch.Tensor, seq: SequenceBase) -> torch.Tensor:
    seq_id_str = str(seq.seq_id)
    if seq.has_to_cache_block:
        last_logits = logits[seq.to_cache_last_token_id]
        self.seq_last_logits_map[seq_id_str] = last_logits
        return last_logits
    # If no cached block, return cached value if available, otherwise use last logit
    if seq_id_str in self.seq_last_logits_map:
        return self.seq_last_logits_map[seq_id_str]
    # Fallback: use last logit from current batch and cache it
    last_logits = logits[-1] if logits.shape[0] > 0 else None
    if last_logits is not None:
        self.seq_last_logits_map[seq_id_str] = last_logits
        return last_logits
    raise ValueError(f"Cannot fetch last logits for sequence {seq.seq_id}: empty logits tensor")
```

### 3. Bounds Checking Fix (`diffulex/strategy/block_diffusion/engine/kvcache_manager.py`)

- **Enhanced bounds validation**: Added check to ensure `prev_block_idx >= 0` in addition to `< seq.num_blocks`
- **Prevents AssertionError**: Fixes assertion failures when `prev_block_idx` is out of bounds

**Before:**
```python
if prev_block_idx < seq.num_blocks:
    token_ids: list[int] = seq.block(prev_block_idx)
```

**After:**
```python
if 0 <= prev_block_idx < seq.num_blocks:
    token_ids: list[int] = seq.block(prev_block_idx)
```

### 4. Test Implementation (`examples/test_async_inference.py`)

Created a comprehensive test file for async inference with fast_dllm_v2:

- **Full async workflow**: Tests the complete async inference pipeline
- **Configurable parameters**: Supports command-line arguments for model path, prompts, tokens, temperature, etc.
- **Performance metrics**: Reports total tokens, time, TPS, and diffusion steps
- **Error handling**: Includes proper error handling and cleanup
- **Follows existing patterns**: Matches the structure of other test files like `test_fastdllmv2_diffulex_gsm8k.py`

## Technical Details

### Thread Safety
- All scheduler operations run in a single-threaded executor to prevent race conditions
- Request addition is synchronous to maintain scheduler state consistency
- Model inference runs asynchronously in the executor for non-blocking execution

### Async Benefits
- Non-blocking inference: Allows other async tasks to run during model inference
- Better resource utilization: Can handle concurrent requests more efficiently
- Integration ready: Compatible with async frameworks (FastAPI, aiohttp, etc.)

### Backward Compatibility
- Synchronous `generate()` method remains unchanged
- All existing functionality preserved
- No breaking changes to the API

## Testing

The async feature has been tested with:
- Fast_dLLM_v2 model with block_diffusion strategy
- Multiple prompts in batch
- Various sampling parameters (temperature, max_tokens)
- Error scenarios and edge cases

## Files Changed

1. `diffulex/engine/tp_worker.py` - Async implementation improvements
2. `diffulex/sampler/base.py` - Bug fixes for async execution
3. `diffulex/strategy/block_diffusion/engine/kvcache_manager.py` - Bounds checking fix
4. `examples/test_async_inference.py` - New test file (NEW FILE)

## Usage Example

```python
import asyncio
from diffulex import Diffulex, SamplingParams

# Create worker
worker = Diffulex(
    model="/path/to/model",
    model_name="fast_dllm_v2",
    decoding_strategy="block_diffusion",
    # ... other config
)

# Run async inference
async def run_inference():
    prompts = ["prompt1", "prompt2", "prompt3"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    outputs = await worker.generate_async(prompts, sampling_params)
    return outputs

outputs = asyncio.run(run_inference())
```

## Issues Fixed

1. ✅ **KeyError in sampler**: Fixed `KeyError: 1` when `has_to_cache_block` is False
2. ✅ **AssertionError in sequence**: Fixed bounds checking in `kvcache_manager.py`
3. ✅ **Thread safety**: Ensured scheduler operations are thread-safe in async execution
4. ✅ **String key consistency**: Fixed type mismatch between integer and string keys in sampler map

## Future Improvements

- Consider adding async streaming support for real-time token generation
- Explore batching optimizations for async requests
- Add async support for data-parallel workers (DiffulexDPWorker)
