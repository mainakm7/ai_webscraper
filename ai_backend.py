from fastapi import FastAPI
from pydantic import BaseModel
from scraping_agent import agentexecutor
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(query: Query):
    response = await run_in_threadpool(agentexecutor.invoke, {"input": query.question})
    return {"response": response}
