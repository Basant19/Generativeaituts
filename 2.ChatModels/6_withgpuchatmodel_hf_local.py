from langchain_huggingface import ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
import os

'''change id if this model does not work for you 
'''

# Load tokenizer and model with GPU support
model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Automatically uses GPU if available
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# Create a pipeline with GPU usage
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # 0 = GPU, -1 = CPU
    max_new_tokens=100,
    temperature=0.5
)

# Wrap in LangChain's interface
llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

# Test prompt
result = model.invoke("What is the capital of India?")
print(result.content)
