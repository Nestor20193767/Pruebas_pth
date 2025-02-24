import os
import io
import re
import faiss
import base64
import PyPDF2
import numpy as np
from gtts import gTTS
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import speech_recognition as sr
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from tempfile import NamedTemporaryFile
from PIL import Image
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="COOKIE",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Languages
languages = {"English": "en-US", "Spanish": "es-ES"}

# --- Funciones de extracción de imágenes ---
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
    """Crea un DataFrame con embeddings para imágenes y texto."""
    database = []
    text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    for img_info in images_info:
        # Crear caption y URL
        caption = f"Página {img_info['page']} - Imagen {img_info['img_index']}"
        img_url = get_base64_img(img_info["path"])
        
        # Crear embedding del caption
        caption_embedding = text_model.encode([caption])[0]
        
        # Crear embedding visual (si es necesario)
        try:
            # Decodificar imagen desde base64
            header, encoded = img_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_data))
            
            # Usar modelo de embeddings visuales (ej: CLIP)
            # image_model = SentenceTransformer('clip-ViT-B-32')
            # image_embedding = image_model.encode(image)
            image_embedding = np.zeros(512)  # Placeholder
        except Exception as e:
            image_embedding = np.zeros(512)
        
        database.append({
            "Caption": caption,
            "URL": img_url,
            "Text_Embedding": caption_embedding,
            "Image_Embedding": image_embedding
        })
    
    return pd.DataFrame(database)

# Nueva función para crear índice de imágenes
@st.cache_resource
def create_image_index(image_db):
    """Crea índice FAISS para búsqueda de imágenes."""
    # Usar embeddings de texto por defecto
    embeddings = np.array(image_db['Text_Embedding'].tolist())
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# --- Funciones de procesamiento de texto ---
def read_pdf_in_chunks(file_name, chunk_size=1000):
    """Lee un PDF y lo divide en fragmentos de texto."""
    reader = PyPDF2.PdfReader(file_name)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

@st.cache_resource
def create_embeddings(chunks):
    """Crea embeddings para los fragmentos de texto."""
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = np.array([model.encode(chunk) for chunk in chunks])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model, index

def search_context(model, index, chunks, question, top_k=3):
    """Busca contexto en texto e imágenes."""
    # Búsqueda en texto
    text_context = "\n\n".join(get_text_context(model, index, chunks, question, top_k))
    
    # Búsqueda en imágenes
    image_context = get_image_context(question, top_k=2)
    
    return f"""
    Text Context:
    {text_context}
    
    Image Context:
    {image_context}
    """

def get_text_context(model, index, chunks, question, top_k=3):
    """Obtiene fragmentos de texto relevantes."""
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def get_image_context(question, top_k=2):
    """Obtiene imágenes relevantes usando embeddings."""
    if st.session_state.image_db.empty:
        return "No images available"
    
    # Modelo para embeddings de preguntas
    text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    question_embedding = text_model.encode([question])
    
    # Buscar en el índice
    distances, indices = st.session_state.image_index.search(question_embedding, top_k)
    
    # Obtener URLs
    results = []
    for i in indices[0]:
        row = st.session_state.image_db.iloc[i]
        results.append(f"![{row['Caption']}]({row['URL']})")
    
    return "\n".join(results)

# --- Funciones de audio ---
def transcribe_audio(audio_file, language="en-US"):
    """Transcribe un archivo de audio a texto."""
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

def text_to_speech(text, language='en-US'):
    """Convierte texto a audio."""
    tts = gTTS(text=text, lang=language)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# --- Interfaz de usuario ---
st.sidebar.title("Settings")
gemini_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
st.sidebar.markdown("[Get your GEMINI key](https://aistudio.google.com/app/apikey)")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Inicialización del estado de la sesión
if "COOKIE_voice" not in st.session_state:
    st.session_state.COOKIE_voice = True
if "talk_to_COOKIE" not in st.session_state:
    st.session_state.talk_to_COOKIE = False
if "image_db" not in st.session_state:
    st.session_state.image_db = pd.DataFrame()
if "image_index" not in st.session_state:
    st.session_state.image_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar adicional
with st.sidebar:
    COOKIE_voice = st.checkbox("COOKIE voice", key="COOKIE_voice")
    talk_to_COOKIE = st.checkbox("Talk to COOKIE", key="talk_to_COOKIE")
    option_language = st.radio(
        "COOKIE language",
        ["English", "Spanish"],
        key="language"
    )

# Título principal
st.title("🍪 COOKIE")
st.subheader("Powered by GEMINI")

# Procesamiento del PDF
if gemini_key and uploaded_file:
    # Guardar PDF y procesar contenido
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_pdf.name
    
    # Extraer imágenes
    images_info = extract_images_from_pdf(temp_pdf_path)
    if images_info:
        st.session_state.image_db = create_image_database(images_info)
        st.session_state.image_index = create_image_index(st.session_state.image_db)
        # Limpiar archivos temporales
        for img_info in images_info:
            os.remove(img_info["path"])
    
    # Procesar texto
    chunks = read_pdf_in_chunks(temp_pdf_path)
    model, index = create_embeddings(chunks)
    os.remove(temp_pdf_path)

# Sección de imágenes
if not st.session_state.image_db.empty:
    with st.expander("📸 Document Images Explorer", expanded=True):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_caption = st.selectbox(
                "Select an image from the document:",
                options=st.session_state.image_db['Caption'].tolist()
            )
        
        with col2:
            selected_url = st.session_state.image_db[
                st.session_state.image_db['Caption'] == selected_caption
            ]['URL'].values[0]
            
            st.markdown(f"**Selected Image:** {selected_caption}")
            st.markdown(
                f'<div style="text-align: center; margin: 20px;">'
                f'<img src="{selected_url}" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">'
                f'</div>',
                unsafe_allow_html=True
            )

# Chat principal
st.write("### Ask a Question Based on the Document")

if gemini_key and uploaded_file:
    # Input de texto o audio
    if not st.session_state.talk_to_COOKIE:
        text_question = st.chat_input("Type your question...")
    else:
        audio_file = st.audio_input("Speak your question...")

    # Mostrar historial de chat
    text_box = st.container(height=500)
    
    try:
        with text_box:
            for role, message, *audio in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(message)
                    if role == "assistant" and audio:
                        st.audio(audio[0], format='audio/mpeg', autoplay=True)
                        b64 = base64.b64encode(audio[0].getvalue()).decode()
                        href = f'<a href="data:audio/mpeg;base64,{b64}" download="response.mp3">Download Response Audio</a>'
                        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        pass

    # Procesar pregunta
    question = None
    if audio_file:
        question = transcribe_audio(audio_file, language=languages[option_language])
    elif text_question:
        question = text_question.strip()

    if question:
        # Guardar pregunta en el historial
        st.session_state.chat_history.append(("user", question))
        
        # Buscar contexto y generar respuesta
        context = search_context(model, index, chunks, question)
        prompt = f"""
        Your name is COOKIE, a medical device assistant that answers in the user's language and in a natural way (using emojis).
        You are helping with the device described in the document.
        
        Relevant document context:
        {context}
        
        Provide a clear answer based on the document and include relevant images using markdown syntax: ![Image Description](URL)
        Question: {question}
        """
        
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(contents=prompt)
        response_text = response.text

        # Convertir respuesta a audio
        cleaned_response = re.sub(r'[\U0001F600-\U0001F64F]+', '', response_text)  # Eliminar emojis
        audio_response = text_to_speech(cleaned_response, language=languages[option_language])

        # Guardar respuesta en el historial
        st.session_state.chat_history.append(("assistant", response_text, audio_response))

        # Mostrar respuesta en la UI
        with text_box:
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
