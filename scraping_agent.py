from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentExecutor
from langchain_ollama import ChatOllama
from scaper import webscraper


llama_llm = ChatOllama(model="llama3.1")

scraping_tool = Tool(
    name="web_scraper",
    func=webscraper, 
    description="Use this tool to scrape data from the company website to gather company details."
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

tools = [scraping_tool]
prompt_template = """
Given the following query, decide if it requires scraping a website. 
If it does, use the 'web_scraper' tool to gather information from the provided URL. 
Otherwise, generate a response using the Llama model.
Query: {query}
"""

prompt_structured = PromptTemplate.from_template(prompt_template)
agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llama_llm,
    prompt=prompt_structured,
    verbose=True
)

agentexecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory, 
    handle_parsing_errors=True,
    
)