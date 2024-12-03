import gradio as gr
import requests

# Function to send the user's input to FastAPI
def get_response(question):
    url = "http://localhost:8000/ask"
    payload = {"question": question, "thread_id": "1"}
    response = requests.post(url, json=payload)
    if response.status_code == 201:
        return response.json()
    else:
        return "Error: Unable to reach the backend."

# Create a Gradio interface
def chatbot_interface(question):
    ai_response = get_response(question)
    return ai_response

# Create the Gradio interface with a text input and output
iface = gr.Interface(fn=chatbot_interface, 
                     inputs="text", 
                     outputs="text", 
                     title="SalarySe AI Chatbot", 
                     description="Ask questions about SalarySe", 
                     allow_flagging="never")

# Launch the interface
iface.launch()
