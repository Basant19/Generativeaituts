from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# LLM for chat
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample documents
docs = [
    Document(page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.", metadata={"team": "Royal Challengers Bangalore"}),
    Document(page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.", metadata={"team": "Mumbai Indians"}),
    Document(page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.", metadata={"team": "Chennai Super Kings"}),
    Document(page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.", metadata={"team": "Mumbai Indians"}),
    Document(page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.", metadata={"team": "Chennai Super Kings"}),
]

# Vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# Add documents
vector_store.add_documents(docs)

# ✅ View stored data (formatted)
print("\n📚 Stored Documents:")
stored = vector_store.get(include=['documents', 'metadatas'])
for i, (doc, meta) in enumerate(zip(stored['documents'], stored['metadatas']), 1):
    print(f"{i}. {doc}\n   ➤ Metadata: {meta}")

# ✅ Similarity Search
print("\n🔍 Top 2 Similar Documents (Bowler Related):")
search_results = vector_store.similarity_search(query='Who among these are a bowler?', k=2)
for i, doc in enumerate(search_results, 1):
    print(f"{i}. {doc.page_content}\n   ➤ Team: {doc.metadata['team']}")

# ✅ Similarity Search with Sc
