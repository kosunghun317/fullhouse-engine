"""Small parallel execution helpers for offline tooling."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, Literal, TypeVar


T = TypeVar("T")
R = TypeVar("R")
Backend = Literal["process", "thread"]


def resolve_workers(workers: int | None, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if workers == 0:
        return max(1, min(os.cpu_count() or 1, task_count))
    if workers is None or workers <= 1:
        return 1
    return max(1, min(workers, task_count))


def map_parallel(
    fn: Callable[[T], R],
    tasks: Iterable[T],
    workers: int | None = 1,
    backend: Backend = "process",
) -> list[R]:
    task_list = list(tasks)
    worker_count = resolve_workers(workers, len(task_list))
    if worker_count <= 1:
        return [fn(task) for task in task_list]

    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=worker_count) as pool:
        return list(pool.map(fn, task_list))
