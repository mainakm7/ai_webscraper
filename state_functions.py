from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import TavilySearchResults
from langchain.schema import Document


load_dotenv(find_dotenv())


# llm = ChatOllama(model="llama2:7b", temperature=0.7)
llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
# llm = ChatOllama(model="gemma2:27b", temperature=0.7)

class GraphState(TypedDict):
    datasource: str
    question: str
    generation: str
    web_search: str
    documents: List[str]


db_dir = os.path.join(os.getcwd(), "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata2")

urls = ["https://www.salaryse.com/"]


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [doc for sublist in docs for doc in sublist]


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)


embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")


if not os.path.exists(persistent_directory):
    db = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory=persistent_directory)
else:
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever()


def retrieve(state: GraphState) -> GraphState:
    
    """
    Retrieve documents from the Chroma vectorstore.

    Args:
        state (GraphState): Current state containing the user query.

    Returns:
        GraphState: Updated state with retrieved documents.
    """
    question = state["question"]
    rag_docs = retriever.invoke(question)

    
    retrieved_docs = [doc.page_content for doc in rag_docs]

    updated_state = state.copy()
    updated_state["documents"] = retrieved_docs
    return updated_state 


def generate(state: GraphState) -> GraphState:
    """
    Generate an answer using RAG (Retrieve and Generate) on retrieved documents.

    Args:
        state (GraphState): Current state containing the user query and retrieved documents.

    Returns:
        GraphState: Updated state with a new key `generation` containing the generated response.
    """
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(documents) if documents else "No relevant context available."

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant representing SalarySe, specializing in answering questions about our company, products, and services from our perspective. 
        Speak as if you are part of the company, using "we" to represent SalarySe. 
        Provide clear and concise answers with a maximum of 10 lines. 
        If the information is not available or unclear, respond with "I'm sorry, I don't have that information." 
        Tailor responses to maintain a professional and informative tone.
        
        Question: {question}
        Context: {context}
        Answer:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"]
    )

    rag_chain = prompt | llm | StrOutputParser()

    try:
        generation = rag_chain.invoke({"context": context, "question": question})
    except Exception as e:
        generation = f"I'm sorry, an error occurred while generating the response: {str(e)}"

    # Update the state with the generated response
    updated_state = state.copy()
    updated_state["generation"] = generation.strip()
    return updated_state

    
def grade_documents(state: dict) -> dict:
    """
    Filter out irrelevant documents based on relevance grading.

    Args:
        state (dict): A dictionary containing the current state, including "documents" and "question".

    Returns:
        dict: Updated state with filtered documents and an indication if web search is needed.
    """
    documents = state.get("documents", [])
    question = state.get("question", "")

    
    doc_contents = "\n\n".join(doc for doc in documents)
    web_search = "No"

    
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

    
    retrieval_grader = prompt | llm | JsonOutputParser()

    
    response = retrieval_grader.invoke({"context": doc_contents, "question": question})
    grade = response.get("score", "").lower()

    
    filtered_docs = []
    if grade == "yes":
        filtered_docs.append(doc_contents)
    else:
        web_search = "Yes"

    
    updated_state = state.copy()
    updated_state.update({"documents": filtered_docs, "web_search": web_search})
    return updated_state



def web_search(state: dict) -> dict:
    """
    Retrieve docs from a web search.

    Args:
        state (dict): A dictionary containing the current state, including "question" and "documents".

    Returns:
        dict: Updated state with formatted docs added to the "documents" key and other state data preserved.
    """
    question = state.get("question")
    documents = state.get("documents", [])

    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(question)
    
    search_results = "\n".join([doc["content"] for doc in search_docs])
    web_results = Document(page_content=search_results)
   
    documents.append(web_results)
    doc_contents = "\n\n".join(doc.page_content for doc in documents)

    updated_state = state.copy()
    updated_state["documents"] = doc_contents
    return updated_state


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


def hallucination_check(state: GraphState) -> str:
    """
    Decides if the generated output is hallucinated or grounded in relevant documents.

    Args:
        state (GraphState): Contains `generation`, `documents`, and `question`.

    Returns:
        str: A routed node string indicating one of three states:
             - "useful": The answer is grounded and resolves the question.
             - "not useful": The answer is grounded but does not resolve the question.
             - "not supported": The answer is not grounded in the provided documents.
    """
    generation = state["generation"]
    documents = state["documents"]
    question = state["question"]
    
    combined_docs = "\n\n".join(documents) if documents else "No relevant documents provided."
    
    hallucination_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing whether an answer is grounded in / supported by our context document. 
        Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by our context document. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. user
        Here is the context document:
        \n ------- \n
        {documents}
        \n ------- \n
        Here is the answer: {generation} 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"]
    )

    
    answer_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing whether an answer addresses / resolves a question. 
        Give a binary 'yes' or 'no' score to indicate if the answer resolves the question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. user
        The answer is: {generation}
        The question is: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"]
    )

    hallucination_grader = hallucination_prompt | llm | JsonOutputParser()
    answer_grader = answer_prompt | llm | JsonOutputParser()

    try:
        hallucination_response = hallucination_grader.invoke({"generation": generation, "documents": combined_docs})
        if hallucination_response["score"].lower() == "yes":
            answer_response = answer_grader.invoke({"generation": generation, "question": question})
            if answer_response["score"].lower() == "yes":
                return "useful"
            else:
                return "not useful"
        else:
            return "not supported"
    except Exception as e:
        return f"error: {str(e)}"

  
def route_question(state):
    """
       Decides whether to go to RAG or directly answer using the LLM

       Args: state (GraphState)

       returns: A string to denote the routed node
    """

    question = state["question"]

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert assistant in determining whether to directly answer a user question with the LLM's knowledge 
        or use RAG (Retrieval Augmented Generation) for a more context-based response. 
        Use RAG for questions on specific factual data, document-based knowledge, or retrieval-heavy topics. 
        For open-ended, opinion-based, or general knowledge questions, answer directly with the LLM. 
        If you don't know the question reroute to RAG.
        Return a JSON with a single key 'datasource' and no preamble or explanation. 
        Options are 'rag' or 'direct_answer'. Question to route: {question} 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()

    response = question_router.invoke({"question": question})

    if response["datasource"] == "rag":
        return "vectorstore"
    else:
        return "direct_answer"

    
def generate_direct(state: GraphState) -> GraphState:
    """
    Generate an answer directly by LLM.

    Args:
        state (GraphState): Current state containing the user query and retrieved documents.

    Returns:
        GraphState: Updated state with a new key `generation` containing the generated response.
    """
    question = state["question"]


    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant representing SalarySe, specializing in answering questions about our company, products, and services from our perspective. 
        Speak as if you are part of the company, using "we" to represent SalarySe. 
        Provide clear and concise answers with a maximum of 10 lines. 
        Tailor responses to maintain a professional and informative tone.
        
        Question: {question}
        Answer:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"]
    )

    rag_chain = prompt | llm | StrOutputParser()

    try:
        generation = rag_chain.invoke({"question": question})
    except Exception as e:
        generation = f"I'm sorry, an error occurred while generating the response: {str(e)}"

    # Update the state with the generated response
    updated_state = state.copy()
    updated_state["generation"] = generation.strip()
    return updated_state