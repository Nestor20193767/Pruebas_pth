import streamlit as st
import speech_recognition as sr
import tempfile

st.title("Grabación y Transcripción de Audio con Streamlit")

audio_data = st.audio_input("Graba tu audio aquí")

if audio_data is not None:
    st.audio(audio_data, format="audio/wav")

    # Inicializar el reconocedor de voz
    r = sr.Recognizer()

    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_path = temp_audio_file.name

    # Leer el archivo de audio con SpeechRecognition
    with sr.AudioFile(temp_audio_path) as source:
        audio = r.record(source)

    try:
        # Transcribir el audio a texto
        text = r.recognize_google(audio, language="en-US")  # Especifica el idioma inglés
        st.write("Transcripción:")
        st.write(text)
    except sr.UnknownValueError:
        st.error("No se pudo entender el audio")
    except sr.RequestError as e:
        st.error(f"Error al solicitar el servicio de reconocimiento de voz: {e}")

    # Eliminar el archivo temporal
    import os
    os.remove(temp_audio_path)
