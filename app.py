import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
st.set_page_config(page_title="Election Guide AI", layout="centered")
st.title("🗳️ Smart Election Assistant")

# Set your API Key here or in your environment variables
GEMINI_API_KEY = "AIzaSyDFsfnsj_ORY4LKX_EGQ4zG98CcDAcp5U0"
sbp_v0_af94095dddbd3966274e6b401e69b4e3a7a907a1

sbp_v0_af94095dddbd3966274e6b401e69b4e3a7a907a1

# --- KNOWLEDGE ENGINE ---
@st.cache_resource
def initialize_engine():
    # 1. Load your Election PDF
    loader = PyPDFLoader("election_rules.pdf") 
    data = loader.load()
    
    # 2. Split text for processing
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = splitter.split_documents(data)
    
    # 3. Create Searchable Database using Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vector_store = Chroma.from_documents(docs, embeddings)
    return vector_store

# Initializing logic
try:
    vector_db = initialize_engine()
    retriever = vector_db.as_retriever()
except Exception as e:
    st.warning("Please add 'election_rules.pdf' to your folder and add your API key to start.")
    st.stop()

# --- THE AI ASSISTANT ---
# This model is FREE under Google's 1,000 requests/day tier
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GEMINI_API_KEY,
    temperature=0.2 # Lower temperature = more factual, less "creative"
)

# Strict Guardrails
system_prompt = (
    "You are an expert Election Assistant. Your goal is to help users understand "
    "the voting process, timelines, and steps strictly using the provided documents. "
    "If a user asks about anything else, politely decline. Stay neutral."
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# --- CHAT INTERFACE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_query := st.chat_input("Ask a question about the election process..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # We combine the system prompt and user query
        full_query = f"{system_prompt}\n\nUser Question: {user_query}"
        response = qa_chain.invoke(full_query)
        st.markdown(response["result"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["result"]})