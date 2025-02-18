import os
import faiss
import PyPDF2
import numpy as np
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
from gtts import gTTS
import io
import base64

# Sidebar for API Key and PDF upload
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

genai.configure(api_key=api_key)

# Function to read PDF and split into chunks
def read_pdf_in_chunks(file_name, chunk_size=1000):
    reader = PyPDF2.PdfReader(file_name)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Function to create embeddings
@st.cache_resource
def create_embeddings(chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = np.array([model.encode(chunk) for chunk in chunks])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model, index

# Search function for context
def search_context(model, index, chunks, question, top_k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    return "\n\n".join(chunks[i] for i in indices[0])

# Text-to-speech conversion
def text_to_speech(text, language='en-US'):
    tts = gTTS(text=text, lang=language)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Load and process PDF if uploaded
if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    chunks = read_pdf_in_chunks("uploaded.pdf")
    model, index = create_embeddings(chunks)

# Assistant Interface
st.title('🤖 DIVI')
st.subheader("Powered by GEMINI")
if 'chunks' in locals():
    audio_file = st.audio_input("Speak your question...")

    if audio_file:
        question = transcribe_audio(audio_file)
        
        if question:
            st.chat_message("user").markdown(question)
            
            context = search_context(model, index, chunks, question)
            prompt = f"""
            Your name is Divi, a medical device assistant.
            Relevant document context:
            {context}
            Question:
            {question}
            Provide a clear answer based on the context.
            """

            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(contents=prompt)

            # Remove '*' and replace ':' with new lines before displaying and playing audio
            cleaned_response = response.text.replace('*', ' ').replace(':', '\n')
            st.markdown(cleaned_response)

            audio_response = text_to_speech(cleaned_response)
            st.audio(audio_response, format='audio/mpeg', autoplay=True)

            b64 = base64.b64encode(audio_response.getvalue()).decode()
            href = f'<a href="data:audio/mpeg;base64,{b64}" download="response.mp3">Download Response Audio</a>'
            st.markdown(href, unsafe_allow_html=True)

