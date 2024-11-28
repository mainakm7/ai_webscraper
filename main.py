import gradio as gr
import requests
import uvicorn
from ai_backend import app  
from threading import Thread
import time


def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)


FASTAPI_URL = "http://127.0.0.1:8000/ask/"


def call_fastapi(url):
    response = requests.post(FASTAPI_URL, json={"url": url})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: Unable to process the URL"


def start_gradio():
    iface = gr.Interface(
        fn=call_fastapi,                   
        inputs=gr.Textbox(label="Enter Website URL"),  
        outputs=gr.Textbox(label="Generated Response"),  
        title="Web Scraping Assistant",    
        description="Provide a URL to scrape content and generate a response using Llama 3.1 model.",
    )
    iface.launch()

if __name__ == "__main__":
    
    fastapi_thread = Thread(target=run_fastapi)
    fastapi_thread.start()
    time.sleep(2)
    
    gradio_thread = Thread(target=start_gradio)
    gradio_thread.start()
    
    fastapi_thread.join()
    gradio_thread.join()
