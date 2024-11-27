import streamlit as st
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from PIL import Image
import numpy as np
import tempfile

def preprocess_image(image, input_shape, resize_to_256):
    """Redimensiona y normaliza la imagen según la entrada esperada del modelo."""
    if resize_to_256:
        image = image.resize((256, 256))  # Redimensionar a 256x256 si está activado
    else:
        if len(input_shape) != 4:
            raise ValueError("El modelo espera una entrada con 4 dimensiones (batch, canal, altura, anchura).")
        image = image.resize((input_shape[2], input_shape[3]))  # Dimensiones del modelo
    image_array = np.array(image).astype("float32") / 255.0  # Normalizar entre 0 y 1
    if len(input_shape) == 4 and input_shape[1] == 3:  # RGB
        image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
    image_array = np.expand_dims(image_array, axis=0)  # Agregar batch size
    return image_array

def postprocess_output(output_tensor):
    """Convierte el tensor de salida en una imagen."""
    output_image = output_tensor.squeeze()  # Eliminar dimensiones adicionales
    output_image = np.clip(output_image * 255.0, 0, 255).astype("uint8")  # Desnormalizar
    return Image.fromarray(output_image)

st.title("Procesador de Imágenes con ONNX :)")

# Permitir al usuario cargar un modelo ONNX
uploaded_model = st.file_uploader("Sube tu modelo ONNX", type=["onnx"])

# Permitir al usuario cargar una imagen
uploaded_image = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Control para redimensionar la imagen a 256x256
resize_to_256 = st.checkbox("Redimensionar la imagen a 256x256 antes del procesamiento", value=True)

if uploaded_model and uploaded_image:
    try:
        # Crear un archivo temporal para el modelo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
            temp_file.write(uploaded_model.read())
            temp_model_path = temp_file.name

        # Cargar el modelo ONNX
        with st.spinner("Cargando y validando el modelo..."):
            onnx_model = onnx.load(temp_model_path)
            onnx.checker.check_model(onnx_model)
        
        st.success("Modelo ONNX cargado y validado con éxito!")

        # Verificar las entradas del modelo
        session = ort.InferenceSession(temp_model_path)
        model_inputs = session.get_inputs()

        if not model_inputs:
            st.error("El modelo no tiene entradas definidas.")
        else:
            # Obtener dimensiones de entrada del modelo
            input_name = model_inputs[0].name
            input_shape = model_inputs[0].shape

            # Cargar la imagen
            input_image = Image.open(uploaded_image)
            st.image(input_image, caption="Imagen de entrada", use_column_width=True)

            # Preprocesar la imagen con opción de redimensionar
            image_tensor = preprocess_image(input_image, input_shape, resize_to_256)

            # Ejecutar el modelo
            with st.spinner("Procesando la imagen..."):
                outputs = session.run(None, {input_name: image_tensor})
                output_image = postprocess_output(outputs[0])

            # Mostrar la imagen de salida
            st.image(output_image, caption="Imagen procesada", use_column_width=True)

    except Exception as e:
        st.error(f"Error al procesar: {e}")


