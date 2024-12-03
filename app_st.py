import streamlit as st
import requests
import uvicorn
from threading import Thread
from ai_backend import app
from langchain_core.messages import AIMessage, HumanMessage

FASTAPI_URL = "http://127.0.0.1:8000/ask/"

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Start FastAPI backend in a separate thread
if "fastapi_started" not in st.session_state:
    backend_thread = Thread(target=run_fastapi, daemon=True)
    backend_thread.start()
    st.session_state.fastapi_started = True

# Streamlit interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Chat here")
if user_input:
    try:
        response = requests.post(FASTAPI_URL, json={"question": user_input}, timeout=5)
        response.raise_for_status()
        agent_response = response.json().get("response", "No response found.")
    except requests.exceptions.RequestException as e:
        agent_response = f"Error: {str(e)}"

    st.session_state.chat_history.append({"user": user_input, "bot": agent_response})

# Display chat history
for message in st.session_state.chat_history:
    st.write(f"User: {message['user']}")
    st.write(f"Bot: {message['bot']}")
