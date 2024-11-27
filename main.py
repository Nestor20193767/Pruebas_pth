import streamlit as st
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from PIL import Image
import numpy as np
import tempfile

def preprocess_image(image, input_shape, resize_to_256):
    """Preprocesa la imagen para modelos de rango 4 o 2."""
    if len(input_shape) == 4:
        if resize_to_256:
            image = image.resize((256, 256))
        else:
            image = image.resize((input_shape[2], input_shape[3]))
        image_array = np.array(image).astype("float32") / 255.0
        if input_shape[1] == 3:  # Si se esperan 3 canales (RGB)
            image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
        image_array = np.expand_dims(image_array, axis=0)  # Agregar batch size
    elif len(input_shape) == 2:
        if resize_to_256:
            image = image.resize((256, 256))
        image_array = np.array(image).astype("float32") / 255.0
        image_array = image_array.flatten()  # Convertir a vector 1D
        image_array = np.expand_dims(image_array, axis=0)  # Agregar batch size
    else:
        raise ValueError(f"Formato de entrada no soportado: {input_shape}")
    return image_array

def postprocess_output(output_tensor):
    """Convierte el tensor de salida en una imagen."""
    output_image = output_tensor.squeeze()
    output_image = np.clip(output_image * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(output_image)

st.title("Procesador de Imágenes con ONNX")

uploaded_model = st.file_uploader("Sube tu modelo ONNX", type=["onnx"])
uploaded_image = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

resize_to_256 = st.checkbox("Redimensionar la imagen a 256x256 antes del procesamiento", value=True)

if uploaded_model and uploaded_image:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
            temp_file.write(uploaded_model.read())
            temp_model_path = temp_file.name

        with st.spinner("Cargando y validando el modelo..."):
            session = ort.InferenceSession(temp_model_path)
            model_inputs = session.get_inputs()

        if not model_inputs:
            st.error("El modelo no tiene entradas definidas.")
        else:
            input_name = model_inputs[0].name
            input_shape = model_inputs[0].shape

            st.write(f"Dimensiones esperadas de la entrada: {input_shape}")

            input_image = Image.open(uploaded_image)
            st.image(input_image, caption="Imagen de entrada", use_column_width=True)

            image_tensor = preprocess_image(input_image, input_shape, resize_to_256)

            with st.spinner("Procesando la imagen..."):
                outputs = session.run(None, {input_name: image_tensor})
                output_image = postprocess_output(outputs[0])

            st.image(output_image, caption="Imagen procesada", use_column_width=True)

    except Exception as e:
        st.error(f"Error al procesar: {e}")


