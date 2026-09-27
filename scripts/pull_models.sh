#!/usr/bin/env bash
# Descarrega els models necessaris amb el client "ollama" instal·lat
# en local (fora de Docker). Útil per fer proves ràpides al terminal
# abans de muntar tot l'stack amb Docker Compose.
#
# Ús:
#   ./scripts/pull_models.sh

set -euo pipefail

echo "Descarregant el model de xat (llama3.2:1b)..."
ollama pull llama3.2:1b

echo "Descarregant el model d'embeddings (nomic-embed-text)..."
ollama pull nomic-embed-text

echo "Models llestos. Pots executar 'ollama run llama3.2:1b' per provar-ho."
