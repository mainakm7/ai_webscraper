import requests
import pyttsx3
import gradio as gr
from pyngrok import ngrok
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
ngrok_auth_token = os.environ.get("ngrok_auth_token")
ngrok.set_auth_token(ngrok_auth_token)
public_url = ngrok.connect(7860)
print(f"Public URL: {public_url}")

def get_response(question):
    url = "http://localhost:8000/ask"
    payload = {"question": question, "thread_id": "1"}
    response = requests.post(url, json=payload)
    if response.status_code == 201:
        return response.json()
    else:
        return "Error: Unable to reach the backend."


def chatbot_interface(question):
    return get_response(question)


def play_response_tts(ai_response):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  
    engine.say(ai_response)
    engine.runAndWait()


def chatbot_and_tts(question):
    response = chatbot_interface(question)
    return response  

chatbot = gr.Chatbot()
with gr.Blocks() as iface:
    gr.Markdown("## SalarySe AI Chatbot")
    gr.Markdown("Ask questions about SalarySe and click the button to hear the response.")
    
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="Your Question", placeholder="Tell me about SalarySe", elem_id="input-box")
            input_btn = gr.Button("Submit")
        
        with gr.Column():
            output_box = gr.Textbox(label="Chatbot Response", elem_id="output-box")
            listen_btn = gr.Button("Listen")
    
    
    input_btn.click(chatbot_and_tts, inputs=input_box, outputs=output_box)
    listen_btn.click(play_response_tts, inputs=output_box, outputs=[])


iface.launch(server_port=7860)

