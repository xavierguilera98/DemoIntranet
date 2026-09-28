"""
Ingesta de documents per al RAG.

Llegeix tots els fitxers .md de `data/docs/`, els trosseja en
fragments petits, genera els seus embeddings amb un model d'Ollama
i els guarda en una base de dades vectorial Chroma persistent.

Ús:
    python src/ingest.py

Cal executar-lo una vegada abans de fer servir l'app (o cada cop que
s'afegeixi/canviï un document a data/docs/).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

# Amb documents tan curts com els d'aquesta demo (cap supera els
# 1600 caràcters), trossejar-los NOMÉS els perjudica: separa una
# secció (ex. "## Models explorats") de la frase que li dona
# context (ex. "deep learning", al paràgraf anterior), i cap valor
# de chunk_size ho arregla perquè el splitter sempre talla als
# títols. Pujar chunk_size no fusiona això — només fa que calgui un
# k més alt perquè els dos trossos coincideixin al mateix prompt, i
# un k alt arrossega fragments d'altres documents sense relació,
# cosa que confon un model tan petit (1B) i el fa respondre "no ho
# sé" més sovint (efecte comprovat: amb k=4/chunk_size=800 gairebé
# totes les respostes acabaven sent "no ho sé").
#
# La solució de fons: amb un corpus de documents petits i
# temàticament independents com aquest, cada document JA és la
# unitat semàntica correcta — no cal trossejar-lo. Un chunk_size molt
# més gran que el document més llarg fa que cada .md esdevingui un
# sol fragment sencer.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 0


def carregar_documents():
    """Llegeix tots els .md de data/docs/ com a documents de LangChain."""
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Carregats {len(documents)} documents de {DOCS_DIR}")
    return documents


def trossejar_documents(documents):
    """Divideix els documents en fragments petits (chunks)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    fragments = splitter.split_documents(documents)
    print(f"Generats {len(fragments)} fragments (chunk_size={CHUNK_SIZE})")
    return fragments


def indexar(fragments):
    """Genera els embeddings dels fragments i els guarda a Chroma."""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vectorstore = Chroma.from_documents(
        documents=fragments,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Base de dades vectorial guardada a {CHROMA_PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    documents = carregar_documents()
    fragments = trossejar_documents(documents)
    indexar(fragments)
    print("Ingesta completada.")
