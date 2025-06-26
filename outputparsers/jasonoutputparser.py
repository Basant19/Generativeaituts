from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os


load_dotenv()
hf_token=os.getenv ("HUGGINGFACEHUB_ACCESS_TOKEN")
LLM=HuggingFaceEndpoint (
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model=ChatHuggingFace (llm=LLM)
parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)
