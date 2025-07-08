import streamlit as st
from langchain.schema import HumanMessage,SystemMessage,AIMessage
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


## Streamlit UI
st.set_page_config(page_title="My chatbot ")
st.header("👋👋  I am your chatbot 💬 , Ask me a question ❓ ")

if 'mymessages' not in st.session_state:
    st.session_state['mymessages']=[
        SystemMessage(content="Basant chatbot ")
    ]

# Function to get response from the chat model
def chatmodel_response(question):

    st.session_state['mymessages'].append(HumanMessage(content=question))
    response=model(st.session_state['mymessages'])
    st.session_state['mymessages'].append(AIMessage(content=response.content))
    return response.content

input=st.text_input("Input: ",key="input")



submit_button=st.button("Ask Question 🙋❓")

## If ask button is clicked

if submit_button:
    response=chatmodel_response(input)
    st.subheader(" This is your Response : 👇")
    st.write(response)
