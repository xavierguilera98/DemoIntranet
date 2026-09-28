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

## Registre d'experimentació: fiabilitat de les respostes

**Estat actual: obert, no resolt.** Amb el model petit d'aquesta demo
(`llama3.2:1b`, sense GPU), el xatbot respon bé algunes vegades i
malament unes altres a les mateixes preguntes sobre els mateixos
documents. Aquesta secció no documenta un problema arreglat — documenta
el procés de prova i error, què s'ha après de cada intent, i què quedaria
per provar amb més temps. És exactament el tipus de registre que té
sentit portar quan es testeja un sistema d'IA: no totes les baules es
resolen en un cap de setmana, i saber explicar per què no és tan
valuós com tenir-ho tot perfecte.

**Punt de partida:** `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`, `K_FRAGMENTS=3`.
Símptoma original: la pregunta *"quins models de deep learning es van fer
servir al TFM?"* no es responia, tot i que la informació és als documents.

**Experiment 1 — pujar chunk_size i k (`800`/`4`).** Hipòtesi: el
splitter separa la frase de context ("deep learning") de la llista de
models (secció `## Models explorats`) en fragments diferents; un `k`
més alt hauria de recuperar-los tots dos alhora. Resultat comprovat amb
`--debug`: la hipòtesi sobre el tall era certa (el splitter sempre talla
als títols, independentment de `chunk_size`), però l'efecte net va ser
pitjor — amb un corpus de només ~20 fragments, `k=4` arrossega gairebé
mig corpus a cada pregunta, i el model es va confondre amb tant context
barrejat: gairebé totes les respostes van passar a ser "no ho sé".

**Experiment 2 — indexar cada document sencer (`chunk_size=2000`,
`k=2`) + `temperature=0.0`.** Hipòtesi: amb documents tan curts
(cap supera ~1600 caràcters), no calia trossejar-los — cada `.md` ja és
la unitat semàntica correcta. Resultat reportat: **tampoc no va
funcionar de manera fiable** — el xatbot va deixar de saber respondre
gairebé cap pregunta. No es va aïllar si la causa era el chunking per
document, el `k=2`, la baixada de `temperature` a 0, o una combinació
de totes tres (es van canviar diversos paràmetres alhora, cosa que en
retrospectiva no permet saber quin va ser el determinant — una lliçó
metodològica en si mateixa: **canviar una sola variable per prova**).

**Estat després d'aquest registre:** s'ha tornat als valors de partida
(`chunk_size=500`, `chunk_overlap=100`, `k=3`). `temperature=0.0` es
manté (no s'ha confirmat que sigui la causa del problema de
l'Experiment 2, i en principi hauria de fer les respostes més
consistents, no pitjors — però tampoc s'ha aïllat i comprovat a part).

### Possibles passos següents (no provats encara)

- **Provar-ho amb rigor, una variable cada cop:** fixar un petit joc de
  preguntes amb resposta esperada (les 4 d'exemple de dalt en són un bon
  punt de partida), repetir cada pregunta diverses vegades per configuració,
  i anotar el % d'encerts. Sense això, és fàcil confondre soroll aleatori
  (recordeu la discussió sobre `temperature`) amb un efecte real d'un
  paràmetre.
- **Prefixos d'instrucció per a `nomic-embed-text`:** aquest model
  d'embeddings està documentat per rendir millor si les consultes es
  prefixen amb `"search_query: "` i els documents amb `"search_document: "`
  abans de vectoritzar-los. El codi actual no ho fa — podria ser una causa
  real (i fàcil d'arreglar) de retrieval poc fiable, independent de
  `chunk_size` o `k`.
- **`num_ctx` de `ChatOllama`:** no s'ha comprovat si el context complet
  (documents recuperats + pregunta + prompt) es trunca silenciosament pel
  límit de context per defecte d'Ollama. Fixar `num_ctx` explícitament
  (ex. 4096) descartaria aquesta possibilitat.
- **Suavitzar la instrucció "no t'ho inventis" del prompt:** un model tan
  petit pot estar sobre-aplicant aquesta instrucció i refusant respondre
  fins i tot quan la informació hi és, per excés de cautela. Val la pena
  provar variants del prompt.
- **Un model de xat més gran** (`llama3.2:3b` o similar) si el maquinari ho
  aguanta: els models d'1B són coneguts per ser inconsistents seguint
  instruccions, independentment de si el retrieval és perfecte.
- **Un embedding multilingüe diferent:** `nomic-embed-text` no és
  específicament fort en català; un model d'embeddings amb millor suport
  multilingüe podria millorar la qualitat del retrieval sense tocar cap
  altre paràmetre.

Per investigar qualsevol d'aquests punts, `--debug` continua sent l'eina
per distingir si el problema és de retrieval (el fragment correcte no
s'ha recuperat) o del model (s'ha recuperat bé, però la resposta és
dolenta o inconsistent igualment):

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

Aquest projecte es va muntar com una demo de xatbot + intranet local
