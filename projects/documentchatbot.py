import streamlit as st
from dotenv import load_dotenv
import os

# ✅ LangChain v0.3-compatible imports
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 🔐 Load Hugging Face API Token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# 🚀 Streamlit setup
st.set_page_config(page_title="Document Search Chatbot")
st.header("🔍 Document Search Chatbot 👋👋  I am your chatbot 💬 , Ask me a question ❓ 🤖")

# 📄 Load your document
loader = TextLoader(
    r"D:\Generative_ai_practise\Basic_model_setup\projects\sample1.txt",
    encoding='utf-8'
)
docs = loader.load()

# 🧠 Show document stats
num_total_characters = sum(len(doc.page_content) for doc in docs)
st.write(f"Loaded {len(docs)} documents, averaging {num_total_characters / len(docs):,.0f} characters each.")

# 📌 Set up embeddings (corrected version)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 📦 Vector store with FAISS
docsearch = FAISS.from_documents(docs, embeddings)

# 🤖 Load Mistral model (chat interface)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
model = ChatHuggingFace(llm=llm)

# 🔁 Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=docsearch.as_retriever(),
    chain_type="stuff"
)

# 📥 User input
user_query = st.text_input("Ask me a question:", "")

# 🔍 Trigger response
if st.button("Search"):
    if user_query.strip():
        response = qa_chain.run(user_query)
        st.subheader("Answer:")
        st.write(response)
    else:
        st.warning("Please enter a valid question.")
