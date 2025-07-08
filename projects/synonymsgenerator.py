import streamlit as st
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

class Commaseparatedoutput(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")
    

# Set up Streamlit UI
st.set_page_config(page_title="Synonyms Generator")
st.header("📝 Synonyms Generator 🔄")

# Define the template for the chat prompt
template = "You are a helpful assistant. When the user provides any input, you should generate 5 words synonyms in a comma-separated list."
human_template = "{text}"



# Create the ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template)
])


# Create the chain
chain = chat_prompt | model | Commaseparatedoutput()
# Text input for user query
input_text = st.text_input("Enter a word or phrase:", "")

# Button to trigger the synonym generation
if st.button("Generate Synonyms"):
    # Perform the conversation with the chatbot
    if input_text:
        # Get the chatbot response
        response = chain.invoke({"text": input_text})

        # Display the synonyms
        st.write("Synonyms:", ", ".join(response))
    else:
        st.write("Please enter a word or phrase.")