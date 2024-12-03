from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import TavilySearchResults
from langchain.schema import Document



load_dotenv(find_dotenv())

llm = ChatOllama(model="llama3.1", temperature=0.5)

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str] = []
    

    
db_dir = os.path.join(os.getcwd(), "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata2")

def retrieve(state: GraphState) -> GraphState:
    
    """Retrieve documents from the Chroma vectorstore
    
        Args: state (GraphState)
        
        returns: formatted docs as a list to state["context"]
    
    """
    
    urls = [
    "https://www.salaryse.com/"
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    embeddings=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    
    
    if not os.path.exists(persistent_directory):
            db = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory=persistent_directory)
    else:
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever()
    
    question = state["question"]
    rag_docs = retriever.invoke(state["question"])
    

    return {"documents": rag_docs,"question":question} 


def generate(state: GraphState) -> GraphState:
    
    """
    
    Generate answer using RAG on retrieved documents

    Args:
        state (GraphState)

    Returns:
        GraphState: new key to GraphState :generation
    """
    
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Give a detailed output as a string but limit the response to 5 lines maximum.
        user
        Question: {question}
        Context: {context}
        Answer: 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"]
    )

    rag_chain = prompt | llm | StrOutputParser()
    
    
    generation = rag_chain.invoke({"context":documents, "question":question})
    return {"generation":generation, "documents":documents, "question":question}
    
    

def grade_documents(state: GraphState) -> GraphState:
    
    """
    
    Filter out irrelevant documents

    Args:
        state (GraphState)

    Returns:
        GraphState: new key to GraphState :generation
    """
    
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keywords related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        user
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"]
    )
    
    filtered_docs = []
    web_search = "No"
    retrieval_grader = prompt | llm | JsonOutputParser()
    
    for doc in documents:
        score = retrieval_grader.invoke({"context":doc.page_content, "question":question})
        grade = score["score"]
        
        if grade.lower() == "yes":
            filtered_docs.append(doc)
        else:
            web_search = "Yes"
            continue
    return {"documents":filtered_docs, "question":question, "web_search": web_search}

def web_search(state: GraphState) -> GraphState:
    
    """ Retrieve docs from web search 
    
        Args: state (GraphState)
        
        returns: formatted docs as a list to state["context"]
    
    """
    question = state["question"]
    documents = state["documents"]
    
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])
    search_results = "\n".join([doc["content"] for doc in search_docs])
    web_results = Document(page_content=search_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents,"question":question} 


def decide_to_generate(state):
    
    """
       Decides whether to generate an answer based on RAG documents or do a web-search instead
       
       Args: state (GraphState)
       
       returns: A string to denote the routed node
    """
    
    web_search_value = state["web_search"]
    
    if web_search_value == "No":
        return "generate"
    else:
        return "websearch"


def hallucination_check(state):
    
    """
       Decides if generated output is llm hallucination or feom relevant document
       
       Args: state (GraphState)
       
       returns: A string to denote the routed node
    
    """
    
    generation = state["generation"]
    documents = state["documents"]
    question = state["question"]
    
    #### Hallucination grader
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
    Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. 
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. user
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    
    ### Answer grader
    prompt_answer = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. user
        The answer is: {generation}
        The question is: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>  
        """,
    input_variables=["generation", "question"],
    )
    
    answer_grader = prompt_answer| llm | JsonOutputParser()
    response = hallucination_grader.invoke({"generation": generation,"documents":documents})
    
    if response["score"].lower() == "yes":
        answer_check = answer_grader.invoke({"generation": generation, "question":question})
        if answer_check["score"].lower() == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"
  
def route_question(state):
    
    """
       Decides whether to go to vectorstore or websearch based on user question
       
       Args: state (GraphState)
       
       returns: A string to denote the routed node
    """
    
    question = state["question"]
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert at routing a user question to a vectorstore or web search. 
        Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. 
        You do not need to be stringent with the keywords in the question related to these topics. 
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. 
        Return the a JSON with a single key 'datasource' and no preamble or explanation. Question to route: {question} 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | llm | StrOutputParser()
    
    response = question_router.invoke({"question":question})
    
    if response["datasource"] == "web_search":
        return "websearch"
    else:
        return "vectorstore"
    
