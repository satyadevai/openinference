import os

from google import genai
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://0.0.0.0:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))


def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return f"The current weather in {location} is sunny with a temperature of 25 degrees Celsius."


def run_weather_example() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text="What is the weather like in Boston & new Delhi?")
            ]
        )
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[get_current_weather],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    function_call_parts = []
    for candidate in response.candidates:
        contents.append(candidate.content)
        for part in candidate.content.parts:
            if tool_call := part.function_call:
                function_call_parts.append(
                    types.Part.from_function_response(
                        name=tool_call.name,
                        response={"result": get_current_weather(**tool_call.args)},
                    )
                )
    contents.append(
        types.Content(
            role="user",
            parts=function_call_parts
        )
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[get_current_weather],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    print(response)
    contents.append(response.candidates[0].content)
    print([content.to_json_dict() for content in contents])

if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    run_weather_example()