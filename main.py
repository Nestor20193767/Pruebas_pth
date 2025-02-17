from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from openai import OpenAI

# Variable global requerida
contenido_simens_multix_impac = {}

def obtener_enlaces_pagina(url):
    """Extrae enlaces de una página con manejo de errores"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        return {
            link.text.strip(): urljoin(url, link['href'])
            for element in soup.find_all(class_='pdf')
            for link in element.find_all('a', href=True)
            if link.text.strip()
        }
        
    except Exception as e:
        st.error(f"Error en {url}: {str(e)}")
        return {}

def proceso_secuencial():
    """Versión secuencial para 7 páginas"""
    global contenido_simens_multix_impac
    
    base_url = "https://www.manualslib.com/manual/2987225/Siemens-Healthcare-Multix-Impact-C.html"
    
    for i in range(1, 8):
        url = f"{base_url}?page={i}#manual"
        if resultados := obtener_enlaces_pagina(url):
            contenido_simens_multix_impac.update(resultados)

proceso_secuencial()

#st.write(contenido_simens_multix_impac)

# Crear DataFrame
df = pd.DataFrame(list(contenido_simens_multix_impac.items()), columns=["Section Title", "URL"])
#st.write(df)
# Configuración del chatbot
context = f"""
You are an assistant that knows about X-Ray Multix Impact C from Siemens. If the user asks about the content of the manual, you must show them this table: {df}. 

You **must use the full link** in every response and **never truncate, abbreviate, or replace parts of the URL with "..."**. Each title has a different link, so **never mix them or generalize**.

Additionally, when referring to more information, provide the exact links from the table. Example:
"If you want to know more about ____, here are the relevant sections:  
- [Title1]({df['URL'][0]})  
- [Title2]({df['URL'][1]})  
- [Title3]({df['URL'][2]})"

Ensure that the **entire** link is always displayed. DO NOT cut the links when you display it. 
"""

API_KEY = 'sk-d42bd3f0ecf64fc58e3fab37d7fb6694'
API_URL = 'https://api.deepseek.com/chat/completions'

st.title("Multix Impact Chatbot")

# Cliente de OpenAI
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# Verificar si ya existe el modelo en el estado de la sesión
if "deepseek_model" not in st.session_state:
    st.session_state["deepseek_model"] = "deepseek-chat" # "deepseek-reasoner" 

# Verificar si ya existe el historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Agregar el contexto inicial al historial de mensajes solo en la primera interacción
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({"role": "system", "content": context})

# Mostrar los mensajes previos (excepto el mensaje 'system')
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Procesar la entrada del usuario
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Llamada a la API para generar la respuesta
        stream = client.chat.completions.create(
            model=st.session_state["deepseek_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    
    # Añadir la respuesta del asistente al historial de mensajes
    st.session_state.messages.append({"role": "assistant", "content": response})

