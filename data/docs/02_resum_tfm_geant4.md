# Resum del TFM: deep learning aplicat a simulacions de Geant4

Aquest document resumeix el Treball de Fi de Màster desenvolupat com
a ajudant de recerca en IA al departament de Física de la UPC (grup
ANT), entre març de 2024 i juliol de 2025.

## Context

Geant4 és un programari de simulació de física de partícules molt
utilitzat en investigació, però les simulacions completes són molt
costoses computacionalment. L'objectiu del projecte era explorar si
models de deep learning generatius podien aproximar els resultats
d'aquestes simulacions molt més ràpid.

## Models explorats

- **VAE (Variational Autoencoders):** per aprendre una representació
  comprimida de les distribucions de sortida de les simulacions.
- **Transformers:** per capturar dependències complexes en les
  seqüències de dades generades.
- **Models de difusió:** per generar mostres sintètiques realistes
  que s'assemblessin a la sortida real de Geant4.

Tots els models es van implementar i entrenar amb PyTorch i
TensorFlow.

## Resultat

El treball va concloure amb una nota de 10, validant que aquest tipus
d'aproximació (substituir part d'una simulació física costosa per un
model generatiu entrenat) és viable per accelerar fluxos de treball
de recerca.

## Relació amb aquest projecte

Aquest TFM no té relació directa amb un xatbot ni amb RAG, però
comparteix la base tècnica de fons: entendre com funcionen els
Transformers i els embeddings per sota és el que fa que, en aquest
altre projecte, triar i ajustar un pipeline de RAG no sigui una caixa
negra.
