from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
prompt = PromptTemplate(
    template='Write a summary for the following document - \n {text}',
    input_variables=['text']
)
model = ChatHuggingFace(llm=llm)
loader = PyPDFLoader(r"D:\Generative_ai_practise\Basic_model_setup\rags\documentloaders\dl-curriculum.pdf")
parser = StrOutputParser()
docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser
print(chain.invoke({'text':docs[0].page_content}))