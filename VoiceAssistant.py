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

# Sidebar for API key and file upload
st.sidebar.title("Settings")
gemini_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

st.title("🤖 DIVI")
st.subheader("Powered by GEMINI")

# Function to transcribe audio to text
def transcribe_audio(audio_file, language="en-US"):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during transcription: {e}")
        return None

# Function to convert text to speech
def text_to_speech(text, language='en-US'):
    tts = gTTS(text=text, lang=language)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Function to extract text from a PDF and split it into chunks
def read_pdf_in_chunks(file_name, chunk_size=1000):
    reader = PyPDF2.PdfReader(file_name)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Function to create embeddings for document chunks
@st.cache_resource
def create_embeddings(chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = np.array([model.encode(chunk) for chunk in chunks])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model, index

# Function to retrieve relevant context based on a query
def search_context(model, index, chunks, question, top_k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    return "\n\n".join(chunks[i] for i in indices[0])

if gemini_key and uploaded_file:
    genai.configure(api_key=gemini_key)
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    chunks = read_pdf_in_chunks("uploaded.pdf")
    model, index = create_embeddings(chunks)

    st.write("### Ask a Question Based on the Document")
    audio_file = st.audio_input("Speak your question...")

    if audio_file:
        question = transcribe_audio(audio_file)

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            context = search_context(model, index, chunks, question)

            prompt = f"""
            Your name is DIVI, a medical device assistant. You are helping with the device described in the document.
            
            Relevant document context:
            {context}
            
            Question:
            {question}
            
            Provide a clear answer based on the document.
            """

            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(contents=prompt)

            st.markdown(response.text)

            audio_response = text_to_speech(response.text)
            st.audio(audio_response, format='audio/mpeg', autoplay=True)

            b64 = base64.b64encode(audio_response.getvalue()).decode()
            href = f'<a href="data:audio/mpeg;base64,{b64}" download="response.mp3">Download Response Audio</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.warning("Please enter your Gemini API key and upload a PDF file to continue.")
