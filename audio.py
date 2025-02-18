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

def transcribe_audio(audio_file, language="en-US"):
    """Transcribe un archivo de audio a texto.

    Args:
        audio_file: El archivo de audio a transcribir (debe ser un objeto tipo UploadedFile).
        language: El idioma del audio (por defecto: "en-US").

    Returns:
        El texto transcrito o None si no se pudo transcribir.
        Además, maneja las excepciones y muestra mensajes de error en Streamlit.
    """
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        st.error("No se pudo entender el audio")
        return None
    except sr.RequestError as e:
        st.error(f"Error al solicitar el servicio de reconocimiento de voz: {e}")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado durante la transcripción: {e}")
        return None

def text_to_speech(text, language='en-US'):
    """Convierte texto a voz y devuelve los bytes del audio en memoria.

    Args:
        text: El texto a convertir a voz.
        language: El idioma del texto (por defecto: "en-US").

    Returns:
        Un objeto BytesIO que contiene los bytes del audio.
    """
    tts = gTTS(text=text, lang=language)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

tab1, tab2, tab3= st.tabs(["Speech-Text","text-Speech", "Speech Assistant"])

with tab1:
    st.title("Grabación y Transcripción de Audio con Streamlit")
    audio_file = st.audio_input("Graba tu audio aquí")

    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")

        text = transcribe_audio(audio_file)  # Llama a la función de transcripción

        if text:  # Verifica si la transcripción fue exitosa
            st.write("Transcripción:")
            st.write(text)

with tab2:
    st.title("Texto a Voz con Streamlit y gTTS")

    texto_ingresado = st.text_area("Ingresa el texto que quieres convertir a voz:", height=150)

    if st.button("Generar Audio"):
        if texto_ingresado:
            audio = text_to_speech(texto_ingresado) # Llama a la función de conversión de texto a voz

            st.audio(audio, format='audio/mpeg', autoplay=True)

            b64 = base64.b64encode(audio.getvalue()).decode()
            href = f'<a href="data:audio/mpeg;base64,{b64}" download="audio.mp3">Descargar audio</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Por favor, ingresa un texto para generar el audio.")

with tab3:
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

    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        chunks = read_pdf_in_chunks("uploaded.pdf")
        model, index = create_embeddings(chunks)
