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

# Sidebar for PDF upload and Gemini API key
with st.sidebar:
    st.title("Settings")
    gemini_api_key = st.text_input("Enter Gemini API Key:", type="password")
    uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

    # Function to read and split PDF into chunks
    def read_pdf_chunks(pdf_file, chunk_size=1000):
        reader = PyPDF2.PdfReader(pdf_file)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    # Create embeddings using Sentence-Transformers
    @st.cache_resource
    def create_embeddings(chunks):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = np.array([model.encode(chunk) for chunk in chunks])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return model, index

    # Search for relevant context based on the question
    def search_context(model, index, chunks, query, top_k=3):
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding), top_k)
        return "\n\n".join(chunks[i] for i in indices[0])

    if uploaded_pdf:
        pdf_chunks = read_pdf_chunks(uploaded_pdf)
        embed_model, embed_index = create_embeddings(pdf_chunks)

        st.title('🤖 DIVI')
        st.subheader("powerd with GEMINI")
        user_audio = st.audio_input("Ask your question...")

        if user_audio:
            recognizer = sr.Recognizer()
            with sr.AudioFile(user_audio) as source:
                audio_data = recognizer.record(source)
            
            try:
                user_query = recognizer.recognize_google(audio_data, language="en-US")
                st.markdown(f"**You asked:** {user_query}")

                document_context = search_context(embed_model, embed_index, pdf_chunks, user_query)
                
                prompt = f"""
                You are Divi, a medical device assistant. Answer the user's question based on the provided document context.
                
                Document Context:
                {document_context}
                
                User Question:
                {user_query}
                """
                
                gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                response = gemini_model.generate_content(contents=prompt)

                st.markdown(f"**Divi's Response:** {response.text}")

                audio_response = gTTS(text=response.text, lang='en')
                audio_bytes = io.BytesIO()
                audio_response.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                st.audio(audio_bytes, format='audio/mpeg', autoplay=True)
            except Exception as e:
                st.error(f"Error during transcription or response: {e}")
    else:
        st.warning("Please upload a PDF document to continue.")
else:
    st.warning("Please enter your Gemini API key.")
