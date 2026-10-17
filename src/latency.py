import contextvars
import logging
import time
from contextlib import contextmanager

from fastapi import Request

logger = logging.getLogger("latency")

logging.basicConfig(level=logging.INFO)

_timings: contextvars.ContextVar = contextvars.ContextVar("timings", default=None)
_cache_status: contextvars.ContextVar = contextvars.ContextVar("cache_status", default=None)

@contextmanager
def stage(name: str):
    """Time a block of code and record it under `name` (milliseconds)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings = _timings.get()
        if timings is not None:
            elapsed = (time.perf_counter() - start) * 1000
            # If a stage runs more than once (e.g. two LLM calls), add them up.
            timings[name] = round(timings.get(name, 0) + elapsed, 2)


def set_cache_status(hit: bool):
    """Mark whether this request was served from the Redis cache."""
    holder = _cache_status.get()
    if holder is not None:
        holder["status"] = "HIT" if hit else "MISS"





async def latency_middleware(request: Request, call_next):
    timings: dict = {}
    cache_holder: dict = {"status": "NONE"}
    t_token = _timings.set(timings)
    c_token = _cache_status.set(cache_holder)
 
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        _timings.reset(t_token)
        _cache_status.reset(c_token)
 
    total = round((time.perf_counter() - start) * 1000, 2)
    timings["total"] = total
 
    response.headers["X-Process-Time-ms"] = str(total)
    response.headers["X-Cache"] = cache_holder["status"]
    response.headers["Server-Timing"] = ", ".join(
        f"{k};dur={v}" for k, v in timings.items()
    )
 
    logger.info(
        "%s %s status=%s cache=%s %s",
        request.method,
        request.url.path,
        response.status_code,
        cache_holder["status"],
        " ".join(f"{k}={v}ms" for k, v in timings.items()),
    )
    return response
 