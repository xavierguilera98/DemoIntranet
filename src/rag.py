"""
Pipeline de RAG (Retrieval-Augmented Generation).

Donada una pregunta, recupera els fragments més rellevants de la
base de dades vectorial (Chroma) i els passa com a context al model
de xat (via Ollama) perquè respongui basant-se en aquesta informació
en lloc d'inventar-la.
"""

import os

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

# Nombre de fragments que es recuperen per cada pregunta.
K_FRAGMENTS = 3

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


def formatar_fragments(fragments):
    """Uneix els fragments recuperats en un sol bloc de text pel prompt."""
    return "\n\n---\n\n".join(f.page_content for f in fragments)


def construir_cadena():
    """Construeix la cadena RAG: retrieval -> prompt -> model -> text."""
    vectorstore = carregar_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_FRAGMENTS})

    model = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
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
    # python src/rag.py
    cadena = construir_cadena()
    print("Pipeline de RAG llest. Escriu una pregunta (Ctrl+C per sortir).\n")
    while True:
        pregunta = input("> ")
        resposta = preguntar(cadena, pregunta)
        print(f"\n{resposta}\n")
