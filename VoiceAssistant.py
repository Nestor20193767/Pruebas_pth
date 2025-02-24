import os
import fitz  # pymupdf
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
from PIL import Image

st.set_page_config(
    page_title="COOKIE",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Settings")
gemini_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
multix_impact = "https://marketing.webassets.siemens-healthineers.com/d168e945589c1618/caf5d5af390f/v/f71be8b1483b/siemens-healthineers_XP_medical-Xray-machine_MULTIX-Impact-E.jpg"

if "COOKIE_voice" not in st.session_state:
    st.session_state.COOKIE_voice = True

if "talk_to_COOKIE" not in st.session_state:
    st.session_state.talk_to_COOKIE = False

with st.sidebar:
    COOKIE_voice = st.checkbox("COOKIE voice", key="COOKIE_voice")
    talk_to_COOKIE = st.checkbox("Talk to COOKIE", key="talk_to_COOKIE")
    option_language = st.radio("COOKIE language", ["English", "Spanish"], key="English")

st.title("🍪 COOKIE")
st.subheader("Powered by GEMINI")
st.write("### Ask a Question Based on the Document")


if gemini_key and uploaded_file:
    text_question = st.chat_input("Type your question...")

    text_box = st.container(height=500)
    
    if st.session_state.talk_to_COOKIE:
        audio_file = st.audio_input("Speak your question...")

else:
    st.info('You need to upload a GEMINI key and a document', icon="ℹ️")

languages = {"English": "en-US", "Spanish": "es-ES"}

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def transcribe_audio(audio_file, language="en-US"):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        return text
    except Exception:
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

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = {}

    for page_number in range(len(doc)):
        page = doc[page_number]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_filename = f"{output_folder}/page_{page_number + 1}_img_{img_index}.{ext}"

            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)

            if page_number + 1 not in image_paths:
                image_paths[page_number + 1] = []
            image_paths[page_number + 1].append(image_filename)

    return image_paths

if gemini_key and uploaded_file:
    genai.configure(api_key=gemini_key)

    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    chunks = read_pdf_in_chunks("uploaded.pdf")
    model, index = create_embeddings(chunks)
    image_paths = extract_images_from_pdf("uploaded.pdf")

    question = None
    
    if audio_file:
        question = transcribe_audio(audio_file, language=languages[option_language])

    if text_question:
        question = text_question.strip()

    with text_box:
        if question:
            st.session_state.chat_history.append(("user", question))
            context = search_context(model, index, chunks, question)

            prompt = f"""
            Your name is COOKIE, a medical device assistant that answers in the user's language.
            You are helping with the device described in the document.

            Relevant document context:
            {context}

            Question:
            {question}

            Provide a clear answer based on the document and say in which section and pages the user can find the information that you give them 
            based on the "table of content" in the document.
            If the section contains images, mention them and use them in your response.
            You can show images using links with this form: ![caption](link), here an example:
            
            ![Multix Impact]({multix_impact})

            Also If the user ask you how the device looks like show this image: {multix_impact}
            """

            ai_model = genai.GenerativeModel("gemini-2.0-flash")
            response = ai_model.generate_content(contents=prompt)
            response_text = response.text
            response_cleared = response.text.replace('*', ' ').replace(':', '\n') 

            audio_response = text_to_speech(response_cleared, language=languages[option_language])
            st.session_state.chat_history.append(("assistant", response_text, audio_response))

            with st.chat_message("user"):
                #st.markdown(f"Hola mi bay ![Multix Impact]({multix_impact})")
                st.markdown(question)

            with st.chat_message("assistant"):
                st.markdown(response_text)
                
                if st.session_state.COOKIE_voice:
                    st.audio(audio_response, format='audio/mpeg', autoplay=True)
                    b64 = base64.b64encode(audio_response.getvalue()).decode()
                    href = f'<a href="data:audio/mpeg;base64,{b64}" download="response.mp3">Download Response Audio</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Mostrar imágenes relevantes si hay en la sección
                for page_number, images in image_paths.items():
                    if f"page {page_number}" in response_text.lower():
                        st.write(f"📄 Relevant Images from Page {page_number}:")
                        for img_path in images:
                            img = Image.open(img_path)
                            img = img.resize((img.width // 2, img.height // 2))  # Reduce size by 50%
                            st.image(img, caption=f"Image from Page {page_number}")
