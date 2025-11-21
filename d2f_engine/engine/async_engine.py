import asyncio
from typing import AsyncGenerator, Dict, List, Tuple, Any, DefaultDict
from collections import defaultdict

from d2f_engine.llm import LLM
from d2f_engine.sampling_params import SamplingParams


class AsyncEngine:
    """Async driver with streaming on top of the sync Engine.

    Maintains a single background stepping task and per-sequence subscriber queues.
    """
    def __init__(self, model: str, **kwargs):
        self._engine = LLM(model, **kwargs)
        self._queues: Dict[int, asyncio.Queue] = {}
        self._pending: set[int] = set()
        self._finished: set[int] = set()
        self._bg_task: asyncio.Task | None = None
        self._has_work = asyncio.Event()
        self._closed = False

    async def __aenter__(self):
        if self._bg_task is None:
            self._bg_task = asyncio.create_task(self._run_background())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._closed = True
        self._has_work.set()
        if self._bg_task is not None:
            try:
                await self._bg_task
            except Exception:
                pass
            self._bg_task = None
        # Ensure underlying sync engine is properly closed
        try:
            # LLMEngine registers atexit but provide explicit shutdown
            if hasattr(self._engine, "exit"):
                await asyncio.to_thread(self._engine.exit)
        except Exception:
            pass

    def add_request(self, prompt: str | List[int], sampling_params: SamplingParams) -> int:
        seq_id = self._engine.add_request(prompt, sampling_params)
        # Initialize per-sequence queue and mark as pending
        if seq_id not in self._queues:
            self._queues[seq_id] = asyncio.Queue()
        self._pending.add(seq_id)
        # Wake background loop
        self._has_work.set()
        return seq_id

    async def stream(self, seq_id: int) -> AsyncGenerator[Tuple[List[int], bool], None]:
        """Yield (token_ids, finished) tuples for the given sequence.

        - For causal LMs: yields 1-token deltas until finished. The final yield will have finished=True.
        - For diffusion LMs: yields only once upon completion with the full token list and finished=True.
        """
        if seq_id not in self._queues:
            # Create an empty queue to avoid KeyErrors; background will populate when available
            self._queues[seq_id] = asyncio.Queue()
        q = self._queues[seq_id]
        while True:
            item = await q.get()
            if item is None:
                break
            yield item  # (tokens, finished)

    async def _run_background(self):
        # Single stepping loop shared by all sequences
        while not self._closed:
            if not self._pending:
                # No active work; wait until a request arrives or close signal
                try:
                    await asyncio.wait_for(self._has_work.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                self._has_work.clear()
                continue

            try:
                outputs, _, _, _, deltas = await asyncio.to_thread(self._engine.step)
            except Exception:
                # In case of unexpected failure, finish all pending sequences to unblock consumers
                for sid in list(self._pending):
                    q = self._queues.get(sid)
                    if q is not None:
                        await q.put(([], True))
                        await q.put(None)
                    self._finished.add(sid)
                    self._pending.discard(sid)
                continue

            # Stream per-step deltas (causal LM only)
            if deltas:
                for sid, toks, finished in deltas:
                    if sid in self._pending:
                        q = self._queues.get(sid)
                        if q is not None and toks:
                            await q.put((toks, bool(finished)))
                        if finished:
                            # Mark done; also send sentinel after finishing event
                            self._finished.add(sid)
                            self._pending.discard(sid)
                            if q is not None:
                                await q.put(None)

            # For engines that do not provide deltas (e.g., diffusion), emit on completion
            if outputs:
                for sid, token_ids in outputs:
                    if sid in self._pending and sid not in self._finished:
                        q = self._queues.get(sid)
                        if q is not None:
                            await q.put((token_ids, True))
                            await q.put(None)
                        self._finished.add(sid)
                        self._pending.discard(sid)
    