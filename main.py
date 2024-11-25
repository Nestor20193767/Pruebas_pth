import streamlit as st
import torch
import tempfile
import os

def display_model_summary(model):
    """Extrae y formatea información clave de un modelo PyTorch."""
    # Obtener la estructura del modelo (resumen de capas)
    summary = []
    for name, layer in model.named_children():
        layer_info = {
            "name": name,
            "type": str(layer.__class__.__name__),
            "params": sum(p.numel() for p in layer.parameters())
        }
        summary.append(layer_info)
    return summary

st.title("Visualizador de Modelos PyTorch")

# Permitir al usuario cargar un archivo .pth
uploaded_file = st.file_uploader("Sube tu archivo de modelo (.pth)", type=["pth"])

if uploaded_file:
    try:
        # Crear un archivo temporal para guardar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Cargar el modelo desde el archivo temporal
        with st.spinner("Cargando el modelo..."):
            # Cargar el modelo de PyTorch
            model = torch.load(temp_file_path)
            model.eval()  # Establecer el modelo en modo de evaluación

        # Mostrar un mensaje de éxito
        st.success("Modelo PyTorch cargado con éxito!")

        # Mostrar resumen del modelo
        st.header("Resumen del Modelo")
        model_summary = display_model_summary(model)

        st.subheader("Estructura del Modelo")
        st.json(model_summary)

    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")

