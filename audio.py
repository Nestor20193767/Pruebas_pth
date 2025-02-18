import streamlit as st
import speech_recognition as sr

st.title("Grabación y Transcripción de Audio con Streamlit")

audio_file = st.audio_input("Graba tu audio aquí", type=["audio/wav"])  # Especifica el tipo de audio

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    r = sr.Recognizer()

    try:
        # Usar directamente el UploadedFile con SpeechRecognition
        with sr.AudioFile(audio_file) as source:  # No necesitas crear un archivo temporal
            audio = r.record(source)
        text = r.recognize_google(audio, language="en-US")
        st.write("Transcripción:")
        st.write(text)
    except sr.UnknownValueError:
        st.error("No se pudo entender el audio")
    except sr.RequestError as e:
        st.error(f"Error al solicitar el servicio de reconocimiento de voz: {e}")
    except Exception as e: # Captura excepciones generales para debuggear
        st.error(f"Ocurrió un error inesperado: {e}")
