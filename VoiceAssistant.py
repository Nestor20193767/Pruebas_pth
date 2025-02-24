import streamlit as st
import fitz  # PyMuPDF
import base64
import os
import pandas as pd
import faiss
import numpy as np
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer

# Cargar modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    """Convierte una imagen a base64 para mostrarla en HTML."""
    with open(img_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    ext = img_path.split('.')[-1]
    return f"data:image/{ext};base64,{b64}"

def create_image_database(images_info):
    """Crea un dataframe con captions y URLs de imágenes."""
    database = []
    for img_info in images_info:
        caption = f"Página {img_info['page']} - Imagen {img_info['img_index']}"
        img_url = get_base64_img(img_info["path"])
        database.append({"Caption": caption, "URL": img_url})
    return pd.DataFrame(database)

def generate_embeddings(df):
    """Genera embeddings a partir del dataframe y almacena en FAISS."""
    captions = df["Caption"].tolist()
    
    # Convertir captions en embeddings
    embeddings = embedding_model.encode(captions, convert_to_numpy=True)
    
    # Guardar embeddings en FAISS
    d = embeddings.shape[1]  # Dimensión de los embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return index, captions, embeddings

st.title("Extractor de Imágenes desde PDF con Embeddings")

uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name
    
    st.write("Extrayendo imágenes...")
    images_info = extract_images_from_pdf(temp_pdf_path)
    
    if images_info:
        df_images = create_image_database(images_info)

        # Generar embeddings y almacenarlos
        index, captions, embeddings = generate_embeddings(df_images)

        # Sección de consulta con embeddings
        st.subheader("Búsqueda Semántica de Imágenes")
        query_text = st.text_input("Escribe una descripción para buscar imágenes:")

        if query_text:
            query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)
            D, I = index.search(query_embedding, k=1)  # Encuentra la más cercana
            
            closest_caption = captions[I[0][0]]
            closest_url = df_images[df_images['Caption'] == closest_caption]['URL'].values[0]

            st.markdown(f"**Imagen más relevante:** {closest_caption}")
            st.markdown(f"![{closest_caption}]({closest_url})")
        
        st.subheader("Base de Datos Completa")
        st.dataframe(df_images)

    else:
        st.write("No se encontraron imágenes en el PDF.")
    
    os.remove(temp_pdf_path)

