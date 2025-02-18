import streamlit as st
import speech_recognition as sr

tab1, tab2 = st.tabs(["Speech-Text","text-Speech"])
with tab1:
    st.title("Grabación y Transcripción de Audio con Streamlit")
    
    audio_file = st.audio_input("Graba tu audio aquí")  # Especifica el tipo de audio
    
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
with tab2:
    import streamlit as st
    from gtts import gTTS
    import io
    import base64
    
    def texto_a_voz(texto, idioma='en-US'):
        """Convierte texto a voz y devuelve los bytes del audio en memoria."""
        tts = gTTS(text=texto, lang=idioma)
        # En lugar de guardar en un archivo, usamos BytesIO para mantener el audio en memoria
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)  # Importante: regresar al inicio del "archivo"
        return audio_bytes
    
    st.title("Texto a Voz con Streamlit y gTTS")
    
    texto_ingresado = st.text_area("Ingresa el texto que quieres convertir a voz:", height=150)
    
    if st.button("Generar Audio"):
        if texto_ingresado:
            audio = texto_a_voz(texto_ingresado)
    
            # Mostrar el audio en Streamlit
            st.audio(audio, format='audio/mpeg')  # gTTS genera audio en formato MP3
    
            # opción de descarga:
            b64 = base64.b64encode(audio.getvalue()).decode()
            href = f'<a href="data:audio/mpeg;base64,{b64}" download="audio.mp3">Descargar audio</a>'
            st.markdown(href, unsafe_allow_html=True)
    
        else:
            st.warning("Por favor, ingresa un texto para generar el audio.")
