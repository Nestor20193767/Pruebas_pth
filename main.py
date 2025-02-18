# 📌 Import necessary libraries
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import google.generativeai as genai

# 🚀 Configure Gemini API
api_key = "AIzaSyAgDe959MVEgOz7Z5WtXgIIRXY-5DA54co"
client = genai.Client(api_key=api_key)

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

# 🚀 Streamlit Chat Interface
st.title('PDF Q&A with Gemini and FAISS')
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    chunks = read_pdf_in_chunks("uploaded.pdf")
    model, index = create_embeddings(chunks)

    st.chat_message("assistant").markdown("**PDF uploaded and embeddings created! Ask a question:**")

    if question := st.chat_input("Your question..."):
        with st.chat_message("user"):
            st.markdown(question)
        context = search_context(model, index, chunks, question)
        
        # Get answer from Gemini
        prompt = f"""
        Relevant document context:
        {context}

        Question:
        {question}

        Provide a clear answer based on the context.
        """
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response.text)
