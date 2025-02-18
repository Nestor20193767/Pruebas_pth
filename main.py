# 📌 Import necessary libraries
import os
import faiss
import PyPDF2
import numpy as np
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# 🚀 Configure Gemini API
api_key = "AIzaSyAgDe959MVEgOz7Z5WtXgIIRXY-5DA54co"

genai.configure(api_key=api_key)

# 📝 Function to read the PDF and split into chunks
def read_pdf_in_chunks(file_name, chunk_size=1000):
    reader = PyPDF2.PdfReader(file_name)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# ⚙️ Create Embeddings with Sentence-Transformers
@st.cache_resource
def create_embeddings(chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = np.array([model.encode(chunk) for chunk in chunks])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model, index

# 💡 Function to search for relevant context
def search_context(model, index, chunks, question, top_k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    return "\n\n".join(chunks[i] for i in indices[0])

# 🚀 Streamlit App with Sidebar
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    chunks = read_pdf_in_chunks("uploaded.pdf")
    model, index = create_embeddings(chunks)
    st.sidebar.success("PDF uploaded and embeddings created!")

st.title('Xray Multix Impact C Assitant')
if 'chunks' in locals():
    if question := st.chat_input("Ask a question based on the uploaded PDF..."):
        with st.chat_message("user"):
            st.markdown(question)
        context = search_context(model, index, chunks, question)
        
        prompt = f"""
        You are Divi an assitant for medical devices in this case you are helping with the device of the document.
        Relevant document context:
        {context}
        
        Question:
        {question}
        
        Provide a clear answer based on the context.
        """
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(contents=prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response.text)
else:
    st.info("Please upload a PDF to begin chatting.")

