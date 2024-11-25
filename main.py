import streamlit as st
import tensorflow as tf
import onnx
from onnx import numpy_helper
import tempfile

def display_model_summary(model):
    """Convierte el resumen del modelo en texto para mostrarlo en Streamlit."""
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return "\n".join(summary_list)
    """Extrae y formatea información clave de un modelo ONNX."""
    # Obtener el nombre del modelo
    model_name = model.graph.name if model.graph.name else "Sin nombre"

st.title("Visualizador de Modelos (.h5 y .keras)")
    # Extraer información de entradas y salidas
    inputs = [{"name": inp.name, "type": inp.type.tensor_type.elem_type, "shape": [dim.dim_value for dim in inp.type.tensor_type.shape.dim]} for inp in model.graph.input]
    outputs = [{"name": out.name, "type": out.type.tensor_type.elem_type, "shape": [dim.dim_value for dim in out.type.tensor_type.shape.dim]} for out in model.graph.output]

# Permitir al usuario cargar un archivo .h5 o .keras
uploaded_file = st.file_uploader("Sube tu archivo de modelo (.h5 o .keras)", type=["h5", "keras"])
    return model_name, inputs, outputs
st.title("Visualizador de Modelos ONNX")
# Permitir al usuario cargar un archivo .onnx
uploaded_file = st.file_uploader("Sube tu archivo de modelo (.onnx)", type=["onnx"])

if uploaded_file:
    try:
        # Crear un archivo temporal para guardar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Cargar el modelo desde el archivo temporal
        with st.spinner("Cargando el modelo..."):
            model = tf.keras.models.load_model(temp_file_path)
            onnx_model = onnx.load(temp_file_path)
            onnx.checker.check_model(onnx_model)

        # Mostrar un mensaje de éxito
        st.success("Modelo cargado con éxito!")
        st.success("Modelo ONNX cargado y validado con éxito!")
        # Mostrar resumen del modelo
        st.header("Información del Modelo")
        model_name, inputs, outputs = display_model_summary(onnx_model)

        # Mostrar el resumen del modelo
        st.header("Resumen del Modelo")
        summary_text = display_model_summary(model)
        st.text(summary_text)
        st.subheader("Nombre del Modelo")
        st.text(model_name)
        st.subheader("Entradas")
        st.json(inputs)
        st.subheader("Salidas")
        st.json(outputs)

        # Mostrar la configuración del modelo
        st.header("Configuración del Modelo (JSON)")
        st.json(model.to_json())
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
