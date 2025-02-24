import os
import io
import re
import fitz
import faiss
import base64
import PyPDF2
import numpy as np
from gtts import gTTS
import pandas as pd
import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from tempfile import NamedTemporaryFile

st.set_page_config(
    page_title="COOKIE",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar para API key y carga de PDF
st.sidebar.title("Settings")
gemini_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
st.sidebar.markdown("[Get your GEMINI key](https://aistudio.google.com/app/apikey?_gl=1*1a748yk*_ga*MTUyNjgyMjI0NS4xNzQwMTQzOTUx*_ga_P1DBVKWT6V*MTc0MDE0Mzk1MS4xLjAuMTc0MDE0Mzk1MS42MC4wLjE2MTk4ODk4ODY.)")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Inicializar valores en session_state
if "COOKIE_voice" not in st.session_state:
    st.session_state.COOKIE_voice = True 
if "talk_to_COOKIE" not in st.session_state:
    st.session_state.talk_to_COOKIE = False
if "image_db" not in st.session_state:
    st.session_state.image_db = pd.DataFrame()
if "images_embedding_content" not in st.session_state:
    st.session_state.images_embedding_content = {}  # Para almacenar embeddings de imágenes

with st.sidebar:
    COOKIE_voice = st.checkbox("COOKIE voice", key="COOKIE_voice")
    talk_to_COOKIE = st.checkbox("Talk to COOKIE", key="talk_to_COOKIE")
    option_language = st.radio("COOKIE language", ["English", "Spanish"], key="English")
    st.write(st.session_state.COOKIE_voice)

st.title("🍪 COOKIE")
st.subheader("Powered by GEMINI")
st.write("### Ask a Question Based on the Document")

# ------------------ FUNCIONES DE IMÁGENES ------------------
def extract_images_from_pdf(pdf_path):
    """Extrae imágenes de un PDF y las guarda en archivos temporales."""
    doc = fitz.open(pdf_path)
    image_data = []
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = f"extracted_image_{page_num}_{img_index}.{img_ext}"
            with open(img_filename, "wb") as img_file:
                img_file.write(image_bytes)
            image_data.append({
                "page": page_num,
                "img_index": img_index,
                "path": img_filename
            })
    return image_data

def get_base64_img(img_path):
    """Convierte una imagen a formato base64 para mostrarla en HTML."""
    with open(img_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    ext = img_path.split('.')[-1]
    return f"data:image/{ext};base64,{b64}"

def create_image_database(images_info):
    """Crea un DataFrame con las imágenes extraídas."""
    database = []
    for img_info in images_info:
        caption = f"Página {img_info['page']} - Imagen {img_info['img_index']}"
        img_url = get_base64_img(img_info["path"])
        database.append({"Caption": caption, "URL": img_url})
    return pd.DataFrame(database)
# -----------------------------------------------------------

# ------------------ FUNCIONES DE AUDIO Y PDF ------------------
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

def text_to_speech(text, language='en-US'):
    tts = gTTS(text=text, lang=language)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

def read_pdf_in_chunks(file_name, chunk_size=1000):
    reader = PyPDF2.PdfReader(file_name)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

@st.cache_resource
def create_embeddings(chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = np.array([model.encode(chunk) for chunk in chunks])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model, index

def search_context(model, index, chunks, question, top_k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    return "\n\n".join(chunks[i] for i in indices[0])
# -----------------------------------------------------------

# Historial del chat
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

text_box = st.container(height=500)
try:
    with text_box:
        for role, message, *audio in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.markdown(message)
                if role == "assistant" and audio:
                    st.audio(audio[0], format='audio/mpeg', autoplay=True)
                    b64 = base64.b64encode(audio[0].getvalue()).decode()
                    href = f'<a href="data:audio/mpeg;base64,{b64}" download="response.mp3">Download Response Audio</a>'
                    st.markdown(href, unsafe_allow_html=True)
except Exception as e:
    pass

if gemini_key and uploaded_file:
    genai.configure(api_key=gemini_key)
    # Guardar el PDF subido
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Guardar PDF temporalmente para extraer imágenes
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_pdf.name

    # EXTRAER IMÁGENES DEL PDF
    images_info = extract_images_from_pdf(temp_pdf_path)
    if images_info:
        st.session_state.image_db = create_image_database(images_info)
        

    # ----------------- EMBEDDINGS DE IMÁGENES Y ALMACENAMIENTO -----------------
    if not st.session_state.image_db.empty:
        st.subheader("Búsqueda Semántica de Imágenes")
        image_model = SentenceTransformer("all-MiniLM-L6-v2")
        image_captions = st.session_state.image_db["Caption"].tolist()
        image_embeddings = image_model.encode(image_captions, convert_to_numpy=True)
        d = image_embeddings.shape[1]
        image_index = faiss.IndexFlatL2(d)
        image_index.add(image_embeddings)
        # Guardar en session_state para uso futuro
        st.session_state.image_model = image_model
        st.session_state.image_index = image_index
        st.session_state.image_captions = image_captions
        # Almacenar embeddings y captions para incluir en el prompt
        st.session_state.images_embedding_content = {
            "captions": image_captions,
            "embeddings": image_embeddings.tolist()
        }
        
        # Entrada para búsqueda de imágenes
        query_img = st.text_input("Escribe una descripción para buscar imágenes:")
        if query_img:
            query_embedding = image_model.encode([query_img], convert_to_numpy=True)
            distances, indices = image_index.search(query_embedding, k=1)
            closest_caption = image_captions[indices[0][0]]
            closest_url = st.session_state.image_db[st.session_state.image_db["Caption"] == closest_caption]["URL"].values[0]
            st.markdown(f"**Imagen más relevante:** {closest_caption}")
            st.markdown(f"![{closest_caption}]({closest_url})")
    # -------------------------------------------------------------------------

    # Procesamiento de texto y generación de embeddings para el contenido del PDF
    chunks = read_pdf_in_chunks("uploaded.pdf")
    model, index = create_embeddings(chunks)
    
    # Entrada de pregunta: se usa audio o texto según el modo elegido
    if st.session_state.talk_to_COOKIE:
        audio_file = st.audio_input("Speak your question...")
    else:
        text_question = st.chat_input("Type your question...")
    
    question = None
    if st.session_state.talk_to_COOKIE:
        if 'audio_file' in locals() and audio_file:
            languages = {"English": "en-US", "Spanish": "es-ES"}
            question = transcribe_audio(audio_file, language=languages[option_language])
    else:
        if 'text_question' in locals() and text_question:
            question = text_question.strip()

    with text_box:
        if question:
            st.session_state.chat_history.append(("user", question))
            context = search_context(model, index, chunks, question)
            
            # Incorporar en el prompt la información de imágenes almacenada
            images_info_text = "\n".join(st.session_state.images_embedding_content["captions"])
            
            prompt = f"""
            Your name is COOKIE, a medical device assistant that answers in the user's language naturally (and using emojis).
            You are helping with the device described in the document.
            
            Relevant document context:
            {context}
            
            Relevant images information:
            {st.session_state.image_db}
            
            Question:
            {question}
            
            Provide a clear answer based on the document and indicate the section and pages (based on the document's table of content) where the user can find the information.
            you can show the images if is necessary using this format: ![caption](url)
            """
            
            model_gen = genai.GenerativeModel("gemini-2.0-flash")
            response = model_gen.generate_content(contents=prompt)
            response_text = response.text

            # Limpiar respuesta (opcional: remover emojis)
            emoji_pattern = re.compile("[\U0001F600-\U0001F64F"
                                       "\U0001F300-\U0001F5FF"
                                       "\U0001F680-\U0001F6FF"
                                       "\U0001F700-\U0001F77F"
                                       "\U0001F780-\U0001F7FF"
                                       "\U0001F800-\U0001F8FF"
                                       "\U0001F900-\U0001F9FF"
                                       "\U0001FA00-\U0001FA6F"
                                       "\U0001FA70-\U0001FAFF"
                                       "\U00002702-\U000027B0"
                                       "\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)
            cleaned_response = emoji_pattern.sub('', response_text)
            cleaned_response = cleaned_response.replace('*', ' ').replace(':', '\n')
            audio_response = text_to_speech(cleaned_response, language={"English": "en-US", "Spanish": "es-ES"}[option_language])
            
            st.session_state.chat_history.append(("assistant", response_text, audio_response))
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                st.markdown(response_text)
                if st.session_state.COOKIE_voice:
                    st.audio(audio_response, format='audio/mpeg', autoplay=True)
                    b64 = base64.b64encode(audio_response.getvalue()).decode()
                    href = f'<a href="data:audio/mpeg;base64,{b64}" download="response.mp3">Download Response Audio</a>'
                    st.markdown(href, unsafe_allow_html=True)
else:
    st.info('You need to upload a GEMINI key and a document', icon="ℹ️")

