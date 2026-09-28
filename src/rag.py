"""
Pipeline de RAG (Retrieval-Augmented Generation).

Donada una pregunta, recupera els fragments més rellevants de la
base de dades vectorial (Chroma) i els passa com a context al model
de xat (via Ollama) perquè respongui basant-se en aquesta informació
en lloc d'inventar-la.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:1b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

# Nombre de fragments que es recuperen per cada pregunta. Amb pocs
# documents curts com els d'aquesta demo, un k massa baix pot deixar
# fora el fragment correcte si la coincidència semàntica no és
# literal (vegeu mostrar_fragments_recuperats() per diagnosticar-ho).
K_FRAGMENTS = 4

PROMPT_TEMPLATE = """Ets un assistent que respon nomes fent servir el
context proporcionat. Si la resposta no es troba al context, digues
clarament que no ho saps a partir dels documents disponibles — no
t'ho inventis.

Context:
{context}

Pregunta: {question}

Resposta:"""


def carregar_vectorstore():
    """Obre la base de dades vectorial ja indexada per ingest.py."""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


def mostrar_fragments_recuperats(retriever, pregunta: str):
    """Imprimeix quins fragments s'han recuperat per a una pregunta.

    Eina de diagnòstic: si el xatbot no sap respondre alguna cosa que
    sí que hi és als documents, el primer que cal comprovar és si el
    fragment correcte s'ha recuperat. Si no hi és, el problema és de
    retrieval (cal pujar k, o el chunk_size a ingest.py); si hi és
    però la resposta continua sent dolenta, el problema és del model
    de xat (massa petit per seguir el prompt).
    """
    fragments = retriever.invoke(pregunta)
    print(f"\n[debug] {len(fragments)} fragments recuperats per: {pregunta!r}")
    for i, fragment in enumerate(fragments):
        origen = fragment.metadata.get("source", "?")
        print(f"  {i}. ({origen})")
        print(f"     {fragment.page_content[:150].replace(chr(10), ' ')}...")
    print()


def formatar_fragments(fragments):
    """Uneix els fragments recuperats en un sol bloc de text pel prompt."""
    return "\n\n---\n\n".join(f.page_content for f in fragments)


def obtenir_retriever():
    """Construeix el retriever (cerca dels K fragments més semblants)."""
    vectorstore = carregar_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": K_FRAGMENTS})


def construir_cadena(retriever=None):
    """Construeix la cadena RAG: retrieval -> prompt -> model -> text."""
    if retriever is None:
        retriever = obtenir_retriever()

    model = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        # temperature=0 fa que el model sempre triï el token més
        # probable (decodificació "greedy") en lloc de mostrejar-lo.
        # Amb un model tan petit, les probabilitats entre "responc" i
        # "no ho sé" solen estar molt igualades — amb temperature>0
        # (mostreig) això fa que la mateixa pregunta, amb el mateix
        # context recuperat, doni respostes diferents en cada crida.
        # Per a un cas d'ús de RAG/QA (volem consistència, no
        # creativitat), temperature=0 és la pràctica estàndard.
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    cadena = (
        {
            "context": retriever | formatar_fragments,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return cadena


def preguntar(cadena, pregunta: str) -> str:
    """Fa una pregunta a la cadena RAG i retorna la resposta com a text."""
    return cadena.invoke(pregunta)


if __name__ == "__main__":
    # Prova ràpida des del terminal, sense interfície:
    #   python src/rag.py
    # Amb --debug, mostra abans de cada resposta quins fragments
    # s'han recuperat (útil per diagnosticar per què una pregunta no
    # es respon bé):
    #   python src/rag.py --debug
    debug = "--debug" in sys.argv

    retriever = obtenir_retriever()
    cadena = construir_cadena(retriever=retriever)
    print("Pipeline de RAG llest. Escriu una pregunta (Ctrl+C per sortir).\n")
    while True:
        pregunta = input("> ")
        if debug:
            mostrar_fragments_recuperats(retriever, pregunta)
        resposta = preguntar(cadena, pregunta)
        print(f"\n{resposta}\n")
