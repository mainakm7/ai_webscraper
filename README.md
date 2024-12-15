# AI Agent and webscaraper

This repository contains an AI agent built using FastAPI for the backend, Gradio for the frontend, and open-source LLMs like Llama2 or Gemma2. The agent supports functionality such as web search, embedding, and potential deployment on Groq hardware. The project is modular and extensible for various AI workflows.

---

## Folder Structure
```
root
|-- state_functions.py    # Contains LLM definitions and node definition functions
|-- graphbuilder.py       # Contains LangGraph logic for managing the AI graph
|-- ai_app.py             # FastAPI backend implementation
|-- main.py               # Gradio frontend to interact with the backend
```

---

## Requirements

### API Keys
To use the agent effectively, ensure the following API keys are available:
- **Tavily API Key**: Required for web search functionality.
- **Nomic API Key**: Required for generating embeddings.
- **Groq API Key**: Optional, only required if deploying on Groq hardware.

Store these keys in a `.env` file in the root directory:

```
TAVILY_API_KEY=your_tavily_api_key
NOMIC_API_KEY=your_nomic_api_key
GROQ_API_KEY=your_groq_api_key
```

### Python Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```
Ensure that the dependencies include:
- **FastAPI**
- **Gradio**
- **LangGraph**
- **Open-source LLM libraries** (e.g., Llama2 or Gemma2 wrappers)

---
## Model download

Download the opensource models from Ollama. First install Ollama and then use ollama pull from commandline
eg: ollama pull gemma2:27b or ollama pull llama2:70b

---
## How to Run the Project

### Backend: FastAPI Server
The backend is implemented using FastAPI. To start the server:

```bash
uvicorn ai_app:app --reload
```
This will start the server, exposing the endpoint for posting queries.
you can test the endpoint directly from http://server_address:port/docs (Default eg: http://localhost:8000/docs)

### Frontend: Gradio Interface
The frontend is implemented using Gradio. To start the interface:

```bash
python main.py
```
The Gradio app will connect to the FastAPI backend and allow users to post queries.
There is a ngrok interface included for sharing. You will need a ngrok_auth_token for use

---

## Query Format
All queries to the backend should be formatted as a JSON object with a single key:

```json
{
    "question": "Your query here"
}
```

---

## How It Works
1. **Backend (FastAPI)**:
    - The backend runs from `ai_app.py` and provides a REST endpoint to handle AI queries.
    - Queries are routed through the graph defined in `graphbuilder.py`.
    - Functions in `state_functions.py` define the logic for each node in the AI workflow.

2. **Frontend (Gradio)**:
    - The frontend allows users to submit queries via a web interface.
    - Queries are forwarded to the FastAPI endpoint for processing.

3. **LLM Processing**:
    - Queries are processed using open-source LLMs like Llama2 or Gemma2.
    - Tavily is used for web search when external data is needed.
    - Nomic is used for embedding generation.

---

## Configuration for Open-Source LLMs
This project uses open-source LLMs such as Llama2 or Gemma2. Ensure that the necessary weights/models are downloaded and accessible within your runtime environment.

---

## Environment Variables
Ensure all environment variables are set correctly in your `.env` file:

```bash
TAVILY_API_KEY=your_tavily_api_key
NOMIC_API_KEY=your_nomic_api_key
GROQ_API_KEY=your_groq_api_key
```

Use the following Python package to load environment variables:
```bash
pip install python-dotenv
```

---

## Deployment

### Local Deployment
Run both the backend and frontend locally using the steps outlined above.

### Groq Deployment
If you wish to deploy on Groq hardware:
- Add your Groq API key to the `.env` file.
- Modify any hardware-specific configurations in `ai_app.py`.

---

## Future Enhancements
- Extend functionality for more advanced workflows.
- Add support for additional LLMs and APIs.
- Improve frontend interactivity and UX.

---

## License
This project is licensed under the Apache License.

