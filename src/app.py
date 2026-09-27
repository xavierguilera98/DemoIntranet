"""
Interfície de xat mínima amb Streamlit.

Fa d'embolcall visual del pipeline de RAG definit a rag.py: mostra
l'historial de la conversa i envia cada pregunta nova a la cadena.

Ús:
    streamlit run src/app.py
"""

import streamlit as st

from rag import construir_cadena, preguntar

st.set_page_config(page_title="Demo Intranet IA", page_icon="💬")

st.title("💬 Demo: assistent local amb RAG")
st.caption(
    "Xatbot que corre 100% en local (Ollama + Chroma). "
    "Respon només a partir dels documents de data/docs/."
)


@st.cache_resource(show_spinner="Carregant el model i la base de dades vectorial...")
def obtenir_cadena():
    return construir_cadena()


cadena = obtenir_cadena()

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostra l'historial de la conversa
for missatge in st.session_state.historial:
    with st.chat_message(missatge["rol"]):
        st.markdown(missatge["contingut"])

# Camp per a la pregunta nova
pregunta = st.chat_input("Escriu la teva pregunta...")

if pregunta:
    st.session_state.historial.append({"rol": "user", "contingut": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensant..."):
            resposta = preguntar(cadena, pregunta)
        st.markdown(resposta)

    st.session_state.historial.append({"rol": "assistant", "contingut": resposta})
