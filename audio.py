import streamlit as st
import speech_recognition as sr
import io

st.title("Grabación y Transcripción de Audio con Streamlit")

audio_data = st.audio_input("Graba tu audio aquí")

if audio_data is not None:
    st.audio(audio_data, format="audio/wav")

    # Inicializar el reconocedor de voz
    r = sr.Recognizer()

    # Crear un objeto BytesIO a partir de los bytes del audio
    audio_file = io.BytesIO(audio_data)
    audio_file.name = "audio.wav"  # Necesario para SpeechRecognition

    try:
        # Leer el audio directamente desde BytesIO
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        
        # Transcribir el audio a texto
        text = r.recognize_google(audio, language="es-ES")  # Cambiado a español
        st.write("Transcripción:")
        st.write(text)
    except sr.UnknownValueError:
        st.error("No se pudo entender el audio")
    except sr.RequestError as e:
        st.error(f"Error al conectar con el servicio: {e}")
