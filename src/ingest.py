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

# Valors de partida. S'han provat altres combinacions (chunking per
# document sencer, k més alt) sense un resultat clarament millor —
# vegeu la secció "Registre d'experimentació" al README per l'anàlisi
# completa i els propers passos oberts.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


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
