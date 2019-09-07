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

## Descrizione notebook 
Tutti i notebook utilizzati condividono la stessa struttura, l'unica differenza sta nei modelli allenati per ogni notebook. Ogni notebook prevede 4 sezioni. 

La prima è utilizzata per l'import e la preparazione dell'ambiente di lavoro. È possibile indicare alcuni parametri di lavoro e la directory dove salvare i risultati dei training. 


La seconda sezione permette di caricare le immagini di training e validation. È possibile generare nuove immagini a partire dalle originali, tramite i parametri è possibile impostare un seed per la generazione oltre a quante immagini generare. 

La terza sezione prevede la definizioe dei modelli. Per ogni modello va definita una funzione che restituisce una funzione che a sua volta restituisce il modello. Inoltre bisogna definire i parametri di training che dovranno essere utilizzati successivamente.

```python
def phase1_1D64():
  def closure():
    inputs = tf.keras.layers.Input(shape=(width, height, 3), name = "L0_INPUT")
    layer = buildDenseLayer(inputs, layers = 1, size = 64, regularizers = 0.01, flattenInput = True)
    outputs = buildDenseSoftmax(layer)
    return tf.keras.Model(inputs = inputs, outputs = outputs)
  return (closure, "phase1_1D64")
  
  
trInfo = TrainingInfo.getDefaultTPU(
    trainingData,
    trainingLabels,
    validationData,
    validationLabels
)

trInfo.setParameters(
    learningRateList = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7],
    fineTuningIterations = 0,
    mainEpochs = 50,
    batchSize = 128,
    validationFrequency = 1,
    metrics = ['sparse_categorical_accuracy'],
    classWeights = None
)
```

Infine l'ultima parte è destinata al training dei modelli. Una volta configurate le sezione precedenti è sufficiente eseguire la cella.

Nello specifico, per il training, sono stati creati 5 notebook, `ModelBuildStep1` per i primi esperimenti con i primi modelli,`ModelBuildInception`, `ModelBuildResNet`, `ModelBuildCNN` per provare a migliorare i modelli delle rispettive tipologie ed infine `GroupModel` per il training del modello in due step (classificatore del gruppo + 1 classificatore per ogni gruppo). 

È presente il notebook `ModelTest` per testare i modelli allenati.

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

**script/colab/ModelBuilderUtils.py**

Contiene funzioni di supporto per la definizione di layer e blocchi per i modelli keras. Fornisce la classe `Models` dove aggiungere i modelli e i parametri di training tramite la funzione `addModel(modelName, model, trainingInfo)`
 
**script/colab/TrainingUtils.py**

Contiene funzioni di supporto per il training dei modelli. È possibile allenare utilizzando CPU, GPU e TPU. Non tutte le versioni di tensorflow funzionano correttamente. Per utilizzare la CPU e GPU occorre usare una versione uguale o superiore alla 1.14.X, mentre per le TPU la versione 1.13.X (anche se risulta lentissima su colab), oppure la versione 1.14.X (alcune funzionalità non sono state ancora implementate). Fornisce la classe `TrainingInfo` dove è possibile impostare diverse parametri per il training come numero di epoche, learning rate, batch size, ... .

**script/colab/TestUtils.py**

Contiene funzioni di supporto per il testing dei modelli. Permette di calcolare l'accuracy di uno o più modelli e plottare la confusion matrix.
