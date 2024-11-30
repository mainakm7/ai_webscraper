import gradio as gr
import requests
import uvicorn
from ai_backend import app  
from threading import Thread
import time
from scraping_agent import memory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Function to run FastAPI
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)


FASTAPI_URL = "http://127.0.0.1:8000/ask/"


def call_fastapi(query):
    
    response = requests.post(FASTAPI_URL, json={"question": query})
    if response.status_code == 200:
        return response.json().get("response", "No response found.")
    else:
        return "Error: Unable to process the request."



def start_gradio():
    iface = gr.Interface(
        fn=call_fastapi,   
        inputs=gr.Textbox(label="Enter your query or website URL"), 
        outputs=gr.Textbox(label="Generated Response"), 
        title="Web Scraping & AI Assistant",   
        description="Provide a question or URL for scraping. The assistant will generate a response using the Llama 3.1 model or scrape the website content.",  # Description
    )
    iface.launch(share=True)


if __name__ == "__main__":
    
    fastapi_thread = Thread(target=run_fastapi)
    fastapi_thread.start()
    time.sleep(2) 
    
    gradio_thread = Thread(target=start_gradio)
    gradio_thread.start()
    
    fastapi_thread.join()
    gradio_thread.join()

    
    initial_message = "You are an AI assistant that can provide helpful answers using available tools."
    memory.chat_memory.add_message(SystemMessage(content=initial_message))

    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        memory.chat_memory.add_message(HumanMessage(content=user_input))
        
        
        response = requests.post(FASTAPI_URL, json={"question": user_input})
        
        if response.status_code == 200:
            agent_response = response.json().get("response", "No response found.")
            memory.chat_memory.add_message(AIMessage(content=agent_response))
        else:
            memory.chat_memory.add_message(AIMessage(content="Error: Unable to process your query."))
