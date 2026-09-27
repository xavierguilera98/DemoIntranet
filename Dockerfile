FROM python:3.11-slim

# curl fa falta a l'entrypoint per comprovar que Ollama respon i per
# demanar-li que descarregui els models.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh
# Normalitza els finals de línia per si el fitxer arriba amb CRLF
# (típic si es clona a Windows amb core.autocrlf activat) — sense
# això, el shebang "#!/usr/bin/env bash" es trenca dins Linux.
RUN sed -i 's/\r$//' ./scripts/entrypoint.sh \
    && chmod +x ./scripts/entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./scripts/entrypoint.sh"]
