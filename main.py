from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import dotenv_values

from dspy_model import generate, NRequest, NResponse

import uvicorn


def phoenix_debug():
    import phoenix as px

    phoenix_session = px.launch_app()

    from openinference.instrumentation.dspy import DSPyInstrumentor
    from opentelemetry import trace as trace_api
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    endpoint = "http://127.0.0.1:6006/v1/traces"
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))

    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    DSPyInstrumentor().instrument()

    print(phoenix_session.url)


@asynccontextmanager
async def lifespan(app: FastAPI):
   secret = dotenv_values(".secret")
   app.state.secret = secret
   phoenix_debug()
   yield
   print('bye application!')


app = FastAPI(lifespan=lifespan)

@app.get("/")
def I_am_alive():
    return "I am alive!!"

@app.post("/generate", status_code=200)
async def __generate(r: NRequest) -> NResponse:
    try:
        return generate(
            utterance=r.utterance,
            states=r.states,
            current_state=r.current_state,
            previous_state=r.previous_state,
            tasks=r.tasks,
            current_task=r.current_task,
            previous_conversation_history=r.previous_conversation_history,
            node=r.node
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=ex)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# debug mode :-)
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9091,)

