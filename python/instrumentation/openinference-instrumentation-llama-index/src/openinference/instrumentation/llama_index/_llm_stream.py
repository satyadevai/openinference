import inspect
import logging
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, Tuple

from wrapt import FunctionWrapper

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.instrumentation import DispatcherSpanMixin  # type: ignore[attr-defined]
from llama_index.core.multi_modal_llms import MultiModalLLM
from openinference.instrumentation.llama_index._stream import _ResponseStream

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_STREAM_METHODS = ("stream_chat", "stream_complete", "astream_chat", "astream_complete")

# Set by a stream method wrapper for the duration of the call. The span handler adds
# the generator the call returned and its LLM span, so the wrapper can pair them up.
_LLM_STREAMS: "ContextVar[Optional[List[Tuple[Any, Any]]]]" = ContextVar(
    "openinference_llama_index_llm_streams", default=None
)


class _StreamMethodPatcher:
    """
    Wraps the stream methods of LlamaIndex LLM classes.

    A streamed LLM call only sends its request once the returned generator is
    iterated, after the LLM span has exited. The wrapper hands back a generator
    that keeps the LLM span current while it runs, so spans from other
    instrumentors nest under it.
    """

    def __init__(self) -> None:
        self._patched: List[Tuple[type, str, Any, Any]] = []
        self._init_subclass: Optional[Tuple[Any, Any]] = None

    def patch(self) -> None:
        for cls in _subclasses(BaseLLM, MultiModalLLM):
            self._patch_class(cls)
        # LLM classes are often imported after instrumenting.
        original = DispatcherSpanMixin.__dict__["__init_subclass__"]
        patch_class = self._patch_class

        def __init_subclass__(cls: Any, /, **kwargs: Any) -> None:
            original.__get__(None, cls)(**kwargs)
            try:
                patch_class(cls)
            except Exception:
                logger.exception(f"Failed to wrap stream methods of {cls.__qualname__}")

        replacement: Any = classmethod(__init_subclass__)
        setattr(DispatcherSpanMixin, "__init_subclass__", replacement)
        self._init_subclass = (original, replacement)

    def unpatch(self) -> None:
        if self._init_subclass is not None:
            original, replacement = self._init_subclass
            if DispatcherSpanMixin.__dict__.get("__init_subclass__") is replacement:
                setattr(DispatcherSpanMixin, "__init_subclass__", original)
            self._init_subclass = None
        for cls, name, original, patched in reversed(self._patched):
            if cls.__dict__.get(name) is patched:
                setattr(cls, name, original)
        self._patched.clear()

    def _patch_class(self, cls: type) -> None:
        if not issubclass(cls, (BaseLLM, MultiModalLLM)):
            return
        for name in _STREAM_METHODS:
            method = cls.__dict__.get(name)
            if method is None or getattr(method, "__isabstractmethod__", False):
                continue
            wrapper = _async_stream if inspect.iscoroutinefunction(method) else _sync_stream
            patched = FunctionWrapper(method, wrapper)
            setattr(cls, name, patched)
            self._patched.append((cls, name, method, patched))


def _subclasses(*bases: type) -> List[type]:
    seen: Dict[type, None] = {}
    stack: List[type] = list(bases)
    while stack:
        cls = stack.pop()
        for subclass in cls.__subclasses__():
            if subclass not in seen:
                seen[subclass] = None
                stack.append(subclass)
    return list(seen)


def _sync_stream(
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    streams: List[Tuple[Any, Any]] = []
    token = _LLM_STREAMS.set(streams)
    try:
        result = wrapped(*args, **kwargs)
    finally:
        _LLM_STREAMS.reset(token)
    return _with_span_current(result, streams)


async def _async_stream(
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    streams: List[Tuple[Any, Any]] = []
    token = _LLM_STREAMS.set(streams)
    try:
        result = await wrapped(*args, **kwargs)
    finally:
        _LLM_STREAMS.reset(token)
    return _with_span_current(result, streams)


def _with_span_current(result: Any, streams: List[Tuple[Any, Any]]) -> Any:
    for stream, span in streams:
        if stream is result:
            return _ResponseStream(result, span.context, _ended_by_events)
    return result


def _ended_by_events(_: Optional[BaseException]) -> None:
    # Streamed LLM spans end on LlamaIndex's stream events.
    pass
