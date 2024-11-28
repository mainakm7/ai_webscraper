from fastapi import FastAPI
from pydantic import BaseModel
from scraping_agent import agent
from scaper import scrape_static_page, scrape_dynamic_page


app = FastAPI()

class Query(BaseModel):
    question: str
    url: str = None  # Optional URL for scraping

@app.post("/ask/")
async def ask_question(query: Query):
    if query.url:
        # If a URL is provided, scrape the website
        page_title, page_content = scrape_static_page(query.url)  # Or scrape_dynamic_page(query.url)
        return {"answer": f"Page Title: {page_title}\nContent: {page_content[:500]}..."}
    else:
        # Use LangChain (Llama) model for the response
        response = agent.run(query.question)  # LangChain agent processes the query
        return {"answer": response}
