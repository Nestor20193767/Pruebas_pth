import streamlit as st
import tensorflow as tf
import tempfile

def display_model_summary(model):
    """Convierte el resumen del modelo en texto para mostrarlo en Streamlit."""
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return "\n".join(summary_list)

st.title("Visualizador de Modelos H5")

# Permitir al usuario cargar un archivo .h5
uploaded_file = st.file_uploader("Sube tu archivo de modelo (.h5)", type=["h5"])

if uploaded_file:
    try:
        # Crear un archivo temporal para guardar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Cargar el modelo desde el archivo temporal
        with st.spinner("Cargando el modelo..."):
            model = tf.keras.models.load_model(temp_file_path)

        # Mostrar un mensaje de éxito
        st.success("Modelo cargado con éxito!")

        # Mostrar el resumen del modelo
        st.header("Resumen del Modelo")
        summary_text = display_model_summary(model)
        st.text(summary_text)

        # Mostrar la configuración del modelo
        st.header("Configuración del Modelo (JSON)")
        st.json(model.to_json())
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")

