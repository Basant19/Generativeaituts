from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

'''method 1. without chaining and not using external file for prompt template
'''
template =PromptTemplate (
template="""You are a helpful AI assistant specialized in explaining research papers.

Explain the research paper titled: "{paper_input}"
Use an explanation style that is: "{style_input}"
Make sure the explanation is: "{length_input}"

Be clear and accurate. If relevant, include examples or key points from the paper.
""",
input_variables=["paper_input", "style_input", "length_input"],
validate_template=True,  # This ensures that the template is validated before use
#we can use validated parameters to ensure the inputs are valid and all variable are provided this we cannot do in f srting 
)
'''Done without chaining that is why two times we are using invoke method
'''
prompt=template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)