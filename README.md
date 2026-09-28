# Demo Intranet — assistent d'IA local amb RAG

Demo personal que munta, de cap a cap, l'stack típic d'un "xatbot
intern" que corre completament en local: sense que cap prompt ni cap
document surti mai a internet.

No és un producte acabat — és una mostra que demostra que sé muntar
i entenc cadascuna de les peces d'aquest tipus de sistema.

## Per què "local"?

Moltes empreses que treballen amb dades confidencials de client (per
exemple, especificacions tècniques de productes en curs) no poden
enganxar aquesta informació a un ChatGPT o Claude en núvol sense
incomplir acords de confidencialitat. La solució és muntar el propi
"ChatGPT intern": un motor d'inferència que corre en un servidor
propi, amb tota la informació sensible quedant-se sempre dins la
xarxa de l'empresa.

Aquest repositori és una versió petita i personal d'aquest mateix
plantejament.

## Arquitectura

```
Pregunta de l'usuari
        │
        ▼
  ┌───────────┐      recupera fragments      ┌─────────────┐
  │ Streamlit │ ───────────rellevants───────► │    Chroma    │
  │  (xat)    │                               │ (vectorial)  │
  └───────────┘ ◄─────────resposta──────────  └─────────────┘
        │                                            ▲
        │ pregunta + context                         │ embeddings
        ▼                                            │
  ┌────────────────────────────────────────────────────┐
  │                       Ollama                        │
  │   model de xat (llama3.2:1b) + embeddings (nomic)   │
  └────────────────────────────────────────────────────┘
```

| Peça | Eina | Fitxer |
| --- | --- | --- |
| Motor d'inferència | [Ollama](https://ollama.com) | `docker-compose.yml` |
| Orquestració / RAG | LangChain + Chroma | `src/ingest.py`, `src/rag.py` |
| Interfície de xat | Streamlit | `src/app.py` |
| Empaquetat | Docker Compose | `Dockerfile`, `docker-compose.yml` |

## Com arrencar-ho

Amb [Docker](https://www.docker.com/) i Docker Compose instal·lats:

```bash
git clone https://github.com/xavierguilera98/demointranet.git
cd demointranet
docker compose up --build
```

La primera vegada trigarà uns minuts: descarrega els models d'Ollama
i fa la ingesta inicial dels documents de `data/docs/`. Un cop llest,
obre http://localhost:8501.

### Sense Docker (per fer proves ràpides)

```bash
# 1. Instal·la Ollama (https://ollama.com) i descarrega els models
./scripts/pull_models.sh

# 2. Crea un entorn virtual i instal·la les dependències
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Copia la configuració i ajusta-la per a ús local (no Docker)
cp .env.example .env
# canvia OLLAMA_BASE_URL a http://localhost:11434 dins .env

# 4. Fes la ingesta i arrenca l'app
python src/ingest.py
streamlit run src/app.py
```

## Maquinari amb què s'ha provat

Aquesta demo s'ha dissenyat i provat en un portàtil **sense GPU
dedicada** (Intel Core i5-7200U, 8 GB de RAM), per això es fan servir
models petits (1-2B paràmetres). Amb aquest maquinari, les respostes
triguen uns quants segons — en un servidor amb GPU serien gairebé
instantànies.

## Preguntes d'exemple

Amb els documents inclosos a `data/docs/`, pots provar:

- «Quina mida tenen les parcel·les de l'hort?»
- «Quins models de deep learning es van fer servir al TFM?»
- «Quins components fa servir el joc del cronòmetre?»
- «Quina API exposa Ollama?» (hauria de dir que no ho sap — no és
  informació que hi hagi als documents indexats)

## Diagnosticar respostes dolentes (retrieval vs. model)

Quan el xatbot no respon bé, val la pena distingir **on** falla:

- **Falla el retrieval** (el fragment amb la resposta ni s'ha recuperat) → cal
  pujar `K_FRAGMENTS` a `src/rag.py` o `CHUNK_SIZE` a `src/ingest.py`.
- **Falla el model** (el fragment correcte s'ha recuperat, però la resposta
  segueix sent dolenta o inventada) → és una limitació del model de xat
  triat (`llama3.2:1b` és molt petit); caldria un model més gran.

Per veure exactament quins fragments es recuperen per a cada pregunta:

```bash
docker compose exec app python src/rag.py --debug
```

**Important:** si canvies `CHUNK_SIZE` o els documents de `data/docs/`, cal
refer la indexació — l'entrypoint només l'executa si la base vectorial
encara no existeix. Per forçar-ho:

```bash
docker compose down
docker volume ls               # busca el volum "..._chroma_data"
docker volume rm demointranet_chroma_data
docker compose up --build
```

(Això no esborra els models d'Ollama, que viuen en un volum diferent —
no cal tornar-los a descarregar.)

## Què quedaria fora d'aquest MVP

Aquesta demo prioritza mostrar el mecanisme complet abans que
escalar-lo. El pas natural següent, en un context d'empresa real,
inclouria:

- **vLLM** en lloc d'Ollama si calgués servir molts usuaris alhora
  amb una GPU de producció.
- **Autenticació multiusuari** (cada persona amb el seu compte i
  historial), inexistent aquí.
- **LangGraph** per convertir la cadena fixa actual en un agent que
  decideixi entre diverses eines (buscar documents, fer un càlcul,
  cridar una altra API) segons la pregunta.
- Una interfície integrada dins una intranet existent, en lloc d'una
  app Streamlit separada.

## Context

Aquest projecte es va muntar com a preparació per a una entrevista
per a un lloc d'enginyer d'IA junior, on la feina consisteix a
treballar sobre un sistema amb aquest mateix plantejament: IA local
per protegir la confidencialitat de dades de client.
