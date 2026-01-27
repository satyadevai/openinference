from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation import TracerProvider
from openinference.instrumentation.google_adk import GoogleADKInstrumentor


endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

greeting_agent = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash",  # Use a valid model
    instruction="Generate a short, friendly greeting.",
    output_key="last_greeting"  # Save response to state['last_greeting']
)

app_name, user_id, session_id = "state_app", "user123", "session1"
session_service = InMemorySessionService()
runner = Runner(
    agent=greeting_agent,
    app_name=app_name,
    session_service=session_service
)


async def run():
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    print(f"Initial state: {session.state}")
    user_message = Content(parts=[Part(text="Hello")])
    for event in runner.run(user_id=user_id,
                            session_id=session_id,
                            new_message=user_message):
        if event.is_final_response():
            print(f"Agent responded.")

    updated_session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    print(f"State after agent run: {updated_session.state}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
