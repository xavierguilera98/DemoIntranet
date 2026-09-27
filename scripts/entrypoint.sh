#!/usr/bin/env bash
# Entrypoint del contenidor de l'app dins Docker Compose.
#
# 1. Espera que Ollama estigui a punt.
# 2. Descarrega els models de xat i d'embeddings si encara no hi són.
# 3. Fa la ingesta dels documents (data/docs/) si la base vectorial
#    encara no existeix.
# 4. Arrenca la interfície Streamlit.

set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
CHAT_MODEL="${CHAT_MODEL:-llama3.2:1b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
CHROMA_PERSIST_DIR="${CHROMA_PERSIST_DIR:-/data/chroma_db}"

echo "Esperant que Ollama respongui a ${OLLAMA_BASE_URL}..."
until curl -sf "${OLLAMA_BASE_URL}/api/tags" > /dev/null; do
  sleep 2
done
echo "Ollama a punt."

pull_model_si_cal() {
  local model="$1"
  if curl -sf "${OLLAMA_BASE_URL}/api/tags" | grep -q "\"${model}\""; then
    echo "Model ${model} ja descarregat."
  else
    echo "Descarregant ${model} (pot trigar uns minuts la primera vegada)..."
    curl -sf "${OLLAMA_BASE_URL}/api/pull" -d "{\"name\": \"${model}\"}"
  fi
}

pull_model_si_cal "${CHAT_MODEL}"
pull_model_si_cal "${EMBEDDING_MODEL}"

if [ -d "${CHROMA_PERSIST_DIR}" ] && [ -n "$(ls -A "${CHROMA_PERSIST_DIR}" 2>/dev/null)" ]; then
  echo "Base de dades vectorial ja existent a ${CHROMA_PERSIST_DIR}, ometent la ingesta."
else
  echo "Fent la ingesta inicial dels documents..."
  python src/ingest.py
fi

echo "Arrencant la interfície Streamlit..."
exec streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501
