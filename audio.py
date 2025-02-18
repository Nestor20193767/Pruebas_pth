import streamlit as st
import speech_recognition as sr

st.title("Grabación y Transcripción de Audio con Streamlit")

audio_data = st.audio_input("Graba tu audio aquí")

if audio_data is not None:
    st.audio(audio_data, format="audio/wav")

    # Inicializar el reconocedor de voz
    r = sr.Recognizer()

    # Convertir los bytes de audio a un archivo de audio temporal
    with open("audio.wav", "wb") as f:
        f.write(audio_data)

    # Leer el archivo de audio con SpeechRecognition
    with sr.AudioFile("audio.wav") as source:
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

    # Opcional: guardar el audio en un archivo
    # (Ya se guarda automáticamente en audio.wav para la transcripción)
    # with open("audio.wav", "wb") as f:
    #     f.write(audio_data)
    # st.success("Audio guardado exitosamente.")
