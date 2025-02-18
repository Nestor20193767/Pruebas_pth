import streamlit as st
import speech_recognition as sr
import tempfile

st.title("Grabación y Transcripción de Audio con Streamlit")

audio_data = st.audio_input("Graba tu audio aquí")

if audio_data is not None:
    st.audio(audio_data, format="audio/wav")

    # Inicializar el reconocedor de voz
    r = sr.Recognizer()
    audio = r.record(audio_data)

    try:
        # Transcribir el audio a texto
        text = r.recognize_google(audio, language="en-US")  # Especifica el idioma inglés
        st.write("Transcripción:")
        st.write(text)
    except sr.UnknownValueError:
        st.error("No se pudo entender el audio")
    except sr.RequestError as e:
        st.error(f"Error al solicitar el servicio de reconocimiento de voz: {e}")

  
