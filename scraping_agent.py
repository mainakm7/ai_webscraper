from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from llm_definition import LlamaLLM
from scaper import scrape_dynamic_page, scrape_static_page


llama_llm = LlamaLLM()

scraping_tool = Tool(
    name="web_scraper",
    func=scrape_static_page, 
    description="Use this tool to scrape data from a website."
)


tools = [scraping_tool]
prompt_template = """
Given the following query, decide if it requires scraping a website. 
If it does, use the 'web_scraper' tool to gather information from the provided URL. 
Otherwise, generate a response using the Llama model.
Query: {query}
"""


agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llama_llm,
    verbose=True
)
