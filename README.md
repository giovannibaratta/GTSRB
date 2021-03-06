# GTSRB - Attività progettuale computer vision
All'interno di questo repository è possibile trovare i notebook e gli script sviluppati per l'attività progettuale per il corso di Computer vision and Image processing.

È possibile utilizzare i notebook sia attraverso un runtime locale, sia attraverso un runtime remoto. Nel caso si utilizzi [Colab](https://colab.research.google.com/), verrà utilizzato Google Drive come memoria persistente, per questo motivo è necessario caricare tutti gli scrit e i dataset al suo interno. Per poter utilizzare i notebook bisogna creare la seguente gerarchia di cartella e posizionare i dataset generati dagli script all'interno delle cartelle corrette. La gerarchia da creare è la seguente:
  
	  root
	  ├─ script
	  │	└─ colab
	  └─ data
	        ├─ samplingN_WxH
		│	├─ training
		│	└─ validation
		└─ testWxH

All'interno della cartella `data` devono essere inserite le immagini per il training, validation e il testing. Possono essere inseriti più directory, ognuna contenente dei sampling differenti, `N` è l'indice e viene utilizzato all'interno dei notebook per caricare un particolare sampling, `W` è la larghezza delle immagini e `H` l'altezza. Per generare i dati utilizzare gli script ***[dataJoiner.py](https://github.com/giovannibaratta/GTSRB/blob/master/scripts/dataJoiner.py)*** e ***[dataJoinerTestData.py](https://github.com/giovannibaratta/GTSRB/blob/master/scripts/dataJoinerTestData.py])*** contenuti in script. È possibile scaricare il dataset originale sulla quale applicare gli script al seguente link :

[Dataset originale](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)

## Descrizione notebook 
Tutti i notebook utilizzati condividono la stessa struttura, l'unica differenza sta nei modelli allenati per ogni notebook. Ogni notebook prevede 4 sezioni. 

Nella prima sezione vengono effettuati gli import e la preparazione dell'ambiente di lavoro. È possibile indicare alcuni parametri di lavoro e la directory, il tipo di runtime e dove salvare i risultati dei training. 

<img src="https://github.com/giovannibaratta/GTSRB/blob/master/screenshot/section1.PNG" width="450">

La seconda sezione permette di caricare le immagini di training e validation. Tramite le opzioni messe a disposizione nel form è possibile generare nuove immagini a partire dalle originali. È possibile indicare il numero di immagini da generare agendo sia sul valore globale, sia sulla singola classe. 

<img src="https://github.com/giovannibaratta/GTSRB/blob/master/screenshot/section2.PNG" width="450">

> **NOTE**: Se si utilizzano le TPU presenti in Colab, il numero massimo di immagini utilizzabili è limitato. Nel caso di immagini 48x48 il massimo è circa di 80000.

La terza è dedicata alla definizione dei modelli. Per ogni modello va definita una funzione che restituisce una funzione che a sua volta restituisce il modello. Inoltre bisogna definire i parametri di training che dovranno essere utilizzati successivamente.

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

Infine l'ultima parte è dedicata al training dei modelli. Una volta configurate le sezione precedenti è sufficiente eseguire la cella, al termine del training verranno salvati i pesi, dei grafici e il log del training all'interno del percorso indicato.

Nello specifico, per il training, sono stati creati 5 notebook, `ModelBuildStep1` per gli esperimenti con i primi modelli,`ModelBuildInception`, `ModelBuildResNet`, `ModelBuildCNN` per provare a migliorare i modelli delle rispettive tipologie ed infine `GroupModel` per il training del modello in due step (classificatore del gruppo + 1 classificatore per ogni gruppo). 

Con il notebook `ModelTest` è possibile testare i modelli allenati.

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
