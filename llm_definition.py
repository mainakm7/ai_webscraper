from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import LLM

class LlamaLLM(LLM):
    def __init__(self, model_name="meta-llama/Llama-3.1"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _call(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=500, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _identifying_params(self):
        return {"model_name": self.model_name}
