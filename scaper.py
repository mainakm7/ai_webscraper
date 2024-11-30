import os

from dotenv import load_dotenv,find_dotenv

from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())


current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")


def webscraper(query):

    urls = ["https://www.salaryse.com/"]

    loader = RecursiveUrlLoader(urls)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3.1")


    if not os.path.exists(persistent_directory):
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    else:
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    relevant_docs = retriever.get_relevant_documents(query)

    return "\n\n".join([doc.page_content for doc in relevant_docs])