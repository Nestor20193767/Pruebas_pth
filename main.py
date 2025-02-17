import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# -------------------------------
# CONFIGURACIÓN DEL CLIENTE DEEPSEEK
# -------------------------------
API_KEY = st.secrets["DEEPSEEK_API_KEY"]
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# -------------------------------
# FUNCIÓN PARA EXTRAER TEXTO DEL PDF
# -------------------------------
def extraer_texto_pdf(archivo_pdf):
    documento = fitz.open(stream=archivo_pdf.read(), filetype="pdf")
    texto = ""
    for pagina in documento:
        texto += pagina.get_text()
    return texto

# -------------------------------
# FUNCIÓN PARA DIVIDIR EL TEXTO EN SECCIONES
# -------------------------------
def dividir_secciones(texto, longitud=3000):
    """Divide el texto en bloques de longitud fija."""
    return [texto[i:i+longitud] for i in range(0, len(texto), longitud)]

# -------------------------------
# FUNCIÓN PARA CONSULTAR CON DEEPSEEK
# -------------------------------
def consultar_deepseek(pregunta, contexto, modelo="deepseek-chat", max_tokens=500):
    """Consulta el modelo deepseek-chat con la API."""
    respuesta = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": "Responde como un experto en equipos médicos."},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta:\n{pregunta}"}
        ],
        max_tokens=max_tokens,
        temperature=0.5
    )
    return respuesta.choices[0].message['content'].strip()

# -------------------------------
# INTERFAZ DE STREAMLIT
# -------------------------------
st.set_page_config(page_title="Asistente MULTIX Impact C", page_icon="🤖")

st.title("🤖 Asistente MULTIX Impact C (Deepseek)")
st.write("Consulta sobre el equipo de radiografía usando la documentación oficial.")

# Cargar PDF
archivo_pdf = st.file_uploader("📂 Sube el manual de MULTIX Impact C (.pdf)", type=["pdf"])
if archivo_pdf is not None:
    with st.spinner("Extrayendo texto..."):
        texto = extraer_texto_pdf(archivo_pdf)
        secciones = dividir_secciones(texto)

    st.success(f"✅ Texto extraído en {len(secciones)} secciones.")
    st.info("Ahora puedes hacer preguntas usando el contenido del manual.")

    # Entrada de Pregunta
    pregunta = st.text_input("💬 Haz una pregunta sobre el sistema:")
    
    if pregunta:
        with st.spinner("Consultando a Deepseek..."):
            respuestas = []
            for i, sec in enumerate(secciones):
                respuesta = consultar_deepseek(pregunta, sec)
                respuestas.append(f"Sección {i+1}:\n{respuesta}\n")

            resultado = "\n\n".join(respuestas)
            st.subheader("📝 Respuesta:")
            st.text_area("Resultado", value=resultado, height=400)

            # Guardar Respuesta
            if st.button("💾 Guardar Respuesta"):
                with open("respuesta_asistente.txt", "w") as archivo:
                    archivo.write(resultado)
                st.success("✅ Respuesta guardada como 'respuesta_asistente.txt'")

else:
    st.warning("🔺 Esperando que subas el manual en PDF.")

