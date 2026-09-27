# Sobre aquest projecte

Aquest repositori és una demo personal per demostrar com es munta un
assistent d'IA que corre completament en local: un motor d'inferència
(Ollama), una capa de RAG (Retrieval-Augmented Generation) sobre uns
quants documents, una interfície de xat (Streamlit) i tot empaquetat
amb Docker Compose.

## Objectiu

L'objectiu no és crear un producte acabat, sinó mostrar que l'autor
entén i sap muntar de cap a cap l'stack típic d'un "xatbot intern"
d'una empresa que vulgui fer servir IA generativa sense que cap dada
surti mai dels seus propis servidors.

## Abast (MVP)

- Un model de xat petit (1-2B paràmetres) i un model d'embeddings,
  tots dos servits per Ollama.
- Una base de dades vectorial (Chroma) amb un grapat de documents
  d'exemple (els que tens a la carpeta `data/docs/`).
- Un script d'ingesta que trosseja els documents i els indexa.
- Un pipeline de RAG que busca els fragments més rellevants abans
  de demanar la resposta al model.
- Una interfície de xat mínima amb Streamlit.
- Un `docker-compose.yml` que ho aixeca tot amb una sola comanda.

## El que queda fora, expressament

Aquest MVP no inclou autenticació multiusuari, un motor de producció
com vLLM, ni un agent amb LangGraph que decideixi entre diverses
eines. Són el pas natural següent un cop la base funciona, però
sortien de l'abast d'una demo feta en un cap de setmana.
