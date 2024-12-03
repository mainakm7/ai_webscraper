from graphbuilder import agent
from pprint import pprint
from fastapi import FastAPI, status
from pydantic import BaseModel, Field
from typing import Sequence, Dict

app = FastAPI()

class app_input(BaseModel):
    thread_id: str = Field("1")
    question: str = Field("Tell me about SalarySe")

@app.post("/ask", status_code=status.HTTP_201_CREATED)
async def ask_agent(input: app_input):
    responses = []
    config = {"configurable":{"thread_id":input.thread_id}}
    query = {"question": input.question}
    for output in agent.stream(query, config):
        for k, v in output.items():
            print(f"\n finished running: {k}")
            responses.append(v.get("generation", "No generation found"))
    if responses:
        print(responses[-1])
        return responses[-1]
    else:
        return "No output was generated."

