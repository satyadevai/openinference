"""LlamaIndex spans must be current so sibling instrumentors nest under them.

Without attaching the span in new_span, OpenAIInstrumentor (and any user span
opened with start_as_current_span during a LlamaIndex call) starts a new root
trace. See https://github.com/Arize-ai/openinference/issues/4053.
"""

import asyncio
import json
import logging
from typing import Any, Iterator, List, Sequence

import httpx
import pytest
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.llms.openai import OpenAI
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.llama_index._handler import _CurrentSpan
from openinference.instrumentation.openai import OpenAIInstrumentor

dispatcher = get_dispatcher(__name__)


@pytest.fixture
def setup_instrumentation(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()


@pytest.fixture
def with_openai_instrumentation(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OpenAIInstrumentor().uninstrument()
    LlamaIndexInstrumentor().uninstrument()


def test_downstream_span_nests_under_llama_index_span(
    setup_instrumentation: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """A span started during a decorated call should parent under the LlamaIndex span."""

    @dispatcher.span
    def outer() -> None:
        assert trace_api.get_current_span().get_span_context().is_valid
        with tracer_provider.get_tracer(__name__).start_as_current_span("downstream"):
            pass

    outer()

    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    assert "downstream" in spans
    outer_spans = [
        s for s in in_memory_span_exporter.get_finished_spans() if s.name.endswith("outer")
    ]
    assert len(outer_spans) == 1
    outer_span = outer_spans[0]
    child = spans["downstream"]
    assert child.parent is not None
    assert child.parent.span_id == outer_span.context.span_id
    assert child.context.trace_id == outer_span.context.trace_id
    # After the call returns, the LlamaIndex span must not remain current
    # (streaming tests rely on this for unrelated root spans).
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_future_returning_span_is_not_current_after_it_resolves(
    setup_instrumentation: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dispatcher.span
    def schedule() -> "asyncio.Future[int]":
        future = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(future.set_result, 42)
        return future

    assert await schedule() == 42
    with tracer_provider.get_tracer(__name__).start_as_current_span("unrelated"):
        pass

    assert not trace_api.get_current_span().get_span_context().is_valid
    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    assert any(name.endswith("schedule") for name in spans)
    assert spans["unrelated"].parent is None
    assert "Failed to detach context" not in caplog.text


def test_openai_chat_span_shares_trace_with_llama_index_llm(
    with_openai_instrumentation: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """With OpenAIInstrumentor enabled, ChatCompletion nests under OpenAI.chat."""

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Paris."},
                    }
                ],
                "usage": {
                    "prompt_tokens": 19,
                    "completion_tokens": 2,
                    "total_tokens": 21,
                },
            },
        )

    llm = OpenAI(
        model="gpt-4o-mini",
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(answer)),
    )
    llm.chat([ChatMessage(role="user", content="Capital of France?")])

    _assert_nested(in_memory_span_exporter.get_finished_spans(), "OpenAI.chat")


def test_openai_stream_chat_span_shares_trace_with_llama_index_llm(
    with_openai_instrumentation: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The request of a streamed call is sent while iterating, after the call returns."""
    llm = OpenAI(
        model="gpt-4o-mini",
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_stream_answer)),
    )
    chunks = list(llm.stream_chat([ChatMessage(role="user", content="Capital of France?")]))

    assert chunks[-1].message.content == "Paris."
    assert not trace_api.get_current_span().get_span_context().is_valid
    _assert_nested(in_memory_span_exporter.get_finished_spans(), "OpenAI.stream_chat")


async def test_openai_astream_chat_span_shares_trace_with_llama_index_llm(
    with_openai_instrumentation: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    llm = OpenAI(
        model="gpt-4o-mini",
        api_key="sk-test",
        max_retries=0,
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(_stream_answer)),
    )
    stream = await llm.astream_chat([ChatMessage(role="user", content="Capital of France?")])
    chunks = [chunk async for chunk in stream]

    assert chunks[-1].message.content == "Paris."
    assert not trace_api.get_current_span().get_span_context().is_valid
    _assert_nested(in_memory_span_exporter.get_finished_spans(), "OpenAI.astream_chat")


def test_llm_class_defined_after_instrumenting_streams_with_span_current(
    setup_instrumentation: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    class StreamingLLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata()

        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
            return CompletionResponse(text="")

        @llm_completion_callback()
        def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponseGen:
            def gen() -> CompletionResponseGen:
                with tracer.start_as_current_span("request"):
                    yield CompletionResponse(text="hi", delta="hi")

            return gen()

    original = StreamingLLM.__dict__["stream_complete"].__wrapped__
    list(StreamingLLM().stream_complete("hello"))

    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    llm_span = spans["StreamingLLM.stream_complete"]
    request = spans["request"]
    assert request.parent is not None
    assert request.parent.span_id == llm_span.context.span_id

    LlamaIndexInstrumentor().uninstrument()
    assert StreamingLLM.__dict__["stream_complete"] is original


async def test_sync_call_returning_future_restores_previous_span(
    setup_instrumentation: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The exit of a sync call that returns a Future runs later, in a done callback."""
    tracer = tracer_provider.get_tracer(__name__)

    @dispatcher.span
    def returns_future() -> Any:  # not annotated as a Future
        return asyncio.get_running_loop().create_future()

    @dispatcher.span
    def noop() -> None:
        pass

    with caplog.at_level(logging.ERROR), tracer.start_as_current_span("parent") as parent:
        future = returns_future()
        assert trace_api.get_current_span().get_span_context() != parent.get_span_context()
        future.set_result(1)
        assert await future == 1
        await asyncio.sleep(0)
        current = trace_api.get_current_span()
        assert current.get_span_context() == parent.get_span_context()
        assert current.is_recording()
        current.set_attribute("after", "future")
        current.add_event("after future")
        with tracer.start_as_current_span("unrelated"):
            pass

        # The next LlamaIndex span removes what the finished call left behind.
        noop()
        assert trace_api.get_current_span() is parent

        # Repeated calls don't pile up.
        for _ in range(5):
            future = returns_future()
            future.set_result(1)
            await future
            await asyncio.sleep(0)
            assert _attached_depth() <= 1
        noop()
        assert trace_api.get_current_span() is parent

    assert not [r for r in caplog.records if "Failed to detach context" in r.getMessage()]
    finished = in_memory_span_exporter.get_finished_spans()
    spans = {span.name.rpartition(".")[2]: span for span in finished}
    future_spans = [span for span in finished if span.name.endswith("returns_future")]
    assert len(future_spans) == 6
    for future_span in future_spans:
        assert future_span.parent is not None
        assert future_span.parent.span_id == parent.get_span_context().span_id
        assert not future_span.attributes or "after" not in future_span.attributes
        assert not future_span.events
    finished_parent = spans["parent"]
    assert finished_parent.attributes is not None
    assert finished_parent.attributes["after"] == "future"
    assert [event.name for event in finished_parent.events] == ["after future"]
    unrelated = spans["unrelated"]
    assert unrelated.parent is not None
    assert unrelated.parent.span_id == parent.get_span_context().span_id


async def test_sync_call_annotated_to_return_future_is_not_attached(
    setup_instrumentation: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("parent") as parent:

        @dispatcher.span
        def returns_future() -> "asyncio.Future[int]":
            assert trace_api.get_current_span() is parent
            return asyncio.get_running_loop().create_future()

        future = returns_future()
        assert trace_api.get_current_span() is parent
        future.set_result(1)
        assert await future == 1
        await asyncio.sleep(0)
        assert trace_api.get_current_span() is parent

    spans = in_memory_span_exporter.get_finished_spans()
    (future_span,) = [span for span in spans if span.name.endswith("returns_future")]
    assert future_span.parent is not None
    assert future_span.parent.span_id == parent.get_span_context().span_id


def test_span_is_not_current_when_span_setup_fails(
    setup_instrumentation: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    metadata_reads: List[None] = []

    class BrokenMetadataLLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            # Fail only for the span handler, which reads it first; newer
            # llama-index-core also reads it inside the call itself.
            metadata_reads.append(None)
            if len(metadata_reads) == 1:
                raise RuntimeError("metadata unavailable")
            return LLMMetadata()

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
            return CompletionResponse(text="ok")

        def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponseGen:
            raise NotImplementedError

    assert BrokenMetadataLLM().complete("hello").text == "ok"

    assert not trace_api.get_current_span().get_span_context().is_valid
    with tracer_provider.get_tracer(__name__).start_as_current_span("unrelated"):
        pass
    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    assert spans["unrelated"].parent is None


def _stream_answer(request: httpx.Request) -> httpx.Response:
    chunks = [
        {"role": "assistant", "content": "Paris."},
        {},
    ]
    body = "".join(
        "data: "
        + json.dumps(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None if delta else "stop",
                    }
                ],
            }
        )
        + "\n\n"
        for delta in chunks
    )
    return httpx.Response(
        200,
        content=(body + "data: [DONE]\n\n").encode(),
        headers={"content-type": "text/event-stream"},
    )


def _attached_depth() -> int:
    depth, span = 0, trace_api.get_current_span()
    while isinstance(span, _CurrentSpan):
        depth, span = depth + 1, span._self_previous
    return depth


def _assert_nested(spans: Sequence[ReadableSpan], llm_name: str) -> None:
    by_name = {span.name: span for span in spans}
    llm_span = by_name[llm_name]
    completion = by_name["ChatCompletion"]
    assert llm_span.parent is None
    assert completion.parent is not None
    assert completion.parent.span_id == llm_span.context.span_id
    assert completion.context.trace_id == llm_span.context.trace_id
