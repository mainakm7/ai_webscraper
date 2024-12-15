from graphbuilder import agent
from fastapi import FastAPI, status
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

class app_input(BaseModel):
    thread_id: str = Field("1")
    question: str = Field("Tell me about SalarySe")

# @app.post("/ask", status_code=status.HTTP_201_CREATED)
# async def ask_agent(input: app_input):
#     responses = []
#     config = {"configurable":{"thread_id":input.thread_id}}
#     query = {"question": input.question}
#     for output in agent.stream(query, config):
#         for k, v in output.items():
#             print(f"\n finished running: {k}")
#             responses.append(v.get("generation", "No generation found"))
#     if responses:
#         print(responses[-1])
#         return responses[-1]
#     else:
#         return "No output was generated."



@app.post("/ask2", status_code=status.HTTP_201_CREATED)
async def ask_agent(input: app_input):
    responses = []
    config = {"configurable": {"thread_id": input.thread_id}}
    query = {"question": input.question}
    
    def process_stream():
        return [output for output in agent.stream(query, config)]
    
    outputs = await run_in_threadpool(process_stream)

    for output in outputs:
        for k, v in output.items():
            print(f"\nFinished running: {k}")
            responses.append(v.get("generation", "No generation found"))
    
    if responses:
        print(responses[-1])
        return responses[-1]
    else:
        return "No output was generated."