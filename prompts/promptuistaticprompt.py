from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)
st.header("Research Tool")


"""
Without Prompt template
"""
user_input = st.text_input("Ask a question:")
if st.button ('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)

