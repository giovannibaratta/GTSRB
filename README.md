# GTSRB - Attività progettuale computer vision
Contiene i notebook e gli script utilizzati per l'attività progettuale per il corso di computer vision.

Per poter utilizzare i notebook bisogna caricare su google driver le immagini per il training. Bisogna creare la seguente gerarchia di directory.
  
	  root
	  ├─ script
	  │	└─ colab
	  └─ data
	        ├─ test
	        ├─ training
	        └── validation

Per generare i dati utilizzare ***dataJoiner.py*** e ***dataJoinerTestData.py*** contenuti in script.

## Descrizione script

**script/dataJoiner.py**

```sh
dataJoiner.py dirPrefix trainingDir validationDir targetWidth targetHeight [randomSeed] [maxCPU] 
```
Raggruppa le immagini presenti nel dataset in un file per ogni classe (00000,00001, ....). Il dataset iniziale viene suddiviso in training set e validation set contenente rispettivamente il 70% e il 30% delle immagini. Le immagini vengono ridimensionate in base ai valori indicati.

| Parametro | Descrizione |
|---|---|
| dirPrefix | Directory contenente tutte le sottodirectory delle classe (00000,00001,...) |
| trainingDir | Directory di output dove salvare i file di training contenente le immagini. Viene salvato un file per classe |
| validationDir | Directory di output dove salvare i file di validation contenente le immagini. Viene salvato un file per classe |
| targetWidth | Larghezza dell'immagine da utilizzare per il ridimensionamento |
| targetHeight | Altezza dell'immagine da utilizzare per il ridimensionamento |
| randomSeed | Opzionale, valore di default è 0. Utilizzato per campionare le immagini da suddividere tra training e validation set |
| maxCPU | Opzionale (richiede randomSeed se utilizzato), numero massimo di thread da utilizzare in parallelo. |

**script/dataJoinerTestData.py**

```sh
dataJoinerTestData.py inputDir outputDir targetWidth targetHeight labelFilePath 
```
Raggruppa le immagini di testing in un unico file e produce un array numpy contenente le lables. Le immagini vengono ridimensionate in base ai valori indicati.

| Parametro | Descrizione |
|---|---|
| inputDir | Directory contenente tutte le immagini di testing (.ppm) |
| outputDir | Directory di output dove salvare il file contenente le immagini e il file per le labels |
| targetWidth | Larghezza dell'immagine da utilizzare per il ridimensionamento |
| targetHeight | Altezza dell'immagine da utilizzare per il ridimensionamento |
| labelFilePath | File CSV contenente le labels delle immagini di testing |
