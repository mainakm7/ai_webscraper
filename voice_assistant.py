from graphbuilder import agent
import asyncio
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero, deepgram, cartesia
from langchain_ollama import ChatOllama

# Initialize LangChain Ollama model
llama_model = ChatOllama(model="llama3.1", temperature=0.6)


class AssistantLogic:
    def __init__(self):
        self._llama_model = llama_model

    def invoke(self, user_input: str):
        response = agent.invoke({"question": user_input, "thread_id": "1"})
        if response:
            return response
        else:
            return self._llama_model.generate(user_input)

    def on(self, event_name, callback=None):
        def metrics_callback(metrics=None):
            print("Metrics collected:", metrics)
            if callback:
                if metrics is not None:
                    callback(metrics)
                else:
                    callback()
        return metrics_callback

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=("You are an AI assistant representing SalarySe, specializing in answering questions about our company, \
            products, and services from our perspective. Speak as if you are part of the company, using 'we' to represent SalarySe."),
    )
    assistant_logic = AssistantLogic()
    
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),                # Optional, if not using VAD
        stt=deepgram.STT(),                   # Use custom STT object
        llm=assistant_logic,                  # Your custom logic
        tts=cartesia.TTS(),                   # Use custom TTS object
        chat_ctx=initial_ctx                  # Use the initial context dictionary
    )
    
    assistant.start(ctx.room)
    await asyncio.sleep(1)
    await assistant.say("How can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))