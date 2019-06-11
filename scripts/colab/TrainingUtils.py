import os
import tensorflow as tf

class TrainingInfo:
  
  def getDefaultTPU(trainingData, trainingLabels, validationData, validationLabels):
    trInfo = TrainingInfo(trainingData, trainingLabels, validationData, validationLabels)
    trInfo.setParameters(
        batchSize = 1024,
        mainEpochs = 100,
        fineTuningIterations = 4,
        fineTuningEpochs = 25,
        shuffle = True,
        learningRateUpdater = reduceLearningRate(),
        lossFunction=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['sparse_categorical_accuracy']
    )
    return trInfo
  
  def __init__(self,trainingData, trainingLabels, validationData, validationLabels):
    self.__learningRateList = [0.001]
    self.__mainEpochs = 10
    self.__shuffle = True
    self.__fineTuningEpochs = 5
    self.__fineTuningIterations = 3
    self.__batchSize = 64
    self.__weights = None
    self.__trainingData = trainingData
    self.__trainingLabels = trainingLabels
    self.__validationData = validationData
    self.__validationLabels = validationLabels
    self.__freezeLayer = False
    self.__layersToFreeze = -1
    self.__freezeFrom = 1 #1-index
    self.__learningRateUpdater = self.__defaultLearningRateUpdater,
    self.__resumeTraining = None
    self.__metrics = []
    self.__lossFunction = tf.keras.losses.sparse_categorical_crossentropy
    self.__weightsFromFile = None
    
  def __defaultLearningRateUpdater(learningRate, iteration):
    return learningRate
  
  def setParameters(self, 
      batchSize = None,
      learningRateList = None,
      mainEpochs = None,
      fineTuningIterations = None,
      fineTuningEpochs = None,
      shuffleData = None,
      shuffle = None,
      weights = None,
      freezeLayer = None,
      layersToFreeze = None,
      freezeFrom = None,
      learningRateUpdater = None,
      resumeTraining = None,
      lossFunction = None,
      metrics = None,
      weightsFromFile = None,
      ):
    if resumeTraining!= None:
      if len(resumeTraining[1]) < 0 or len(resumeTraining[1]) > len(learningRateList):
        raise ValueError("Resume learning non valido")
      self.__resumeTraining = resumeTraining
    if batchSize != None:
      if batchSize <= 0:
        raise ValueError("Batch size <= 0")
      self.__batchSize = batchSize

    if learningRateList != None:
      if len(learningRateList) <= 0:
        raise ValueError("Learning rate list < 0")
        for lr in learningRateList:
          if lr <= 0:
            raise ValueError("Learning rate < 0")
      self.__learningRateList = learningRateList

    if fineTuningIterations != None:
      if fineTuningIterations < 0:
        raise ValueError("Numero iterationi < 0")
      self.__fineTuningIterations = fineTuningIterations

    if fineTuningEpochs != None:
      if fineTuningEpochs <= 0:
        raise ValueError("Numero fine tuning epoche <= 0")
      self.__fineTuningEpochs = fineTuningEpochs

    if mainEpochs != None:
      if mainEpochs <= 0:
        raise ValueError("Numero epoche <= 0")
      self.__mainEpochs = mainEpochs
  
    if shuffle != None:
      self.__shuffle = shuffle

    if weights != None:
      self.__weights = weights

    if weightsFromFile != None:
      self.__weightsFromFile = weightsFromFile
      
    if freezeLayer != None:
      self.__freezeLayer = freezeLayer
      if self.__freezeLayer == True and self.__freezeFrom > self.__fineTuningIterations:
        self.__freezeFrom = 0
    
    if layersToFreeze != None:
      self.__layersToFreeze = layersToFreeze
    
    if freezeFrom != None:
      if self.__freezeLayer == False:
        raise ValueError("Prima devi impostare l'attributo freeze")
      if freezeFrom < 1:
        raise ValueError("Iterazione dalla quale freezare < 1")
      if self.freezeFrom > self.__fineTuningIterations:
        raise ValueError("Il numero di iterazioni è",self.__fineTuningIterations)
      self.__freezeFrom = freezeFrom

    if learningRateUpdater != None:
      self.__learningRateUpdater = learningRateUpdater
      
    if lossFunction != None:
      self.__lossFunction = lossFunction
    if metrics != None:
      self.__metrics = metrics
      
  def getWeights(self):
    return self.__weights
  def getWeightsFromFile(self):
    return self.__weightsFromFile
  def getLearningRateUpdater(self):
    return self.__learningRateUpdater
  def getResumeTraining(self):
    return self.__resumeTraining
  def getLearningRateList(self):
    return self.__learningRateList
  def getFreezeLayer(self):
    return self.__freezeLayer
  def getFreezeFrom(self):
    return self.__freezeFrom
  def getLayersToFreeze(self):
    return self.__layersToFreeze
  def getMainEpochs(self):
    return self.__mainEpochs
  def getTrainingData(self):
    return self.__trainingData
  def getTrainingLabels(self):
    return self.__trainingLabels
  def getValidationData(self):
    return self.__validationData
  def getValidationLabels(self):
    return self.__validationLabels
  def getShuffle(self):
    return self.__shuffle
  def getBatchSize(self):
    return self.__batchSize
  def getFineTuningIterations(self):
    return self.__fineTuningIterations
  def getFineTuningEpochs(self):
    return self.__fineTuningEpochs
  def getLossFunction(self):
    return self.__lossFunction
  def getMetrics(self):
    return self.__metrics
  
def convertModelToTPU(model, tpu_address):
  return tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
    tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address))
  )
  
def retrieveWeights(model, weightsPath, tpu_address):
  tpu_model = convertModelToTPU(model, tpu_address)
  weightsFile = os.listdir(weightsPath)[0]
  tpu_model.load_weights(weightsPath + weightsFile)
  return tpu_model.get_weights()
  
def trainModel(tpu_address, model, learningRate, trainingData, trainingLabels, validationData, validationLabels,
              epochs, shuffle, batchSize, lossFunction , metrics, callbacks = [], weights = None, weightsFromFile = None, verbose = 1):
  # trasformo il modello da tipo keras a modello compatibile con le TPU
  tpu_model = convertModelToTPU(model, tpu_address)

  # compilo il modello
  tpu_model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate = learningRate),
    loss=lossFunction,
    metrics=metrics
  )

  if weights != None:
    tpu_model.set_weights(weights)
  else:
    if weightsFromFile != None:
      tpu_model.load_weights(weightsFromFile,  by_name =True)
  history = tpu_model.fit(
              x = trainingData,
              y = trainingLabels,
              batch_size = batchSize,
              validation_data = (validationData, validationLabels),
              epochs = epochs,
              shuffle = shuffle,
              verbose = verbose,
              callbacks=callbacks
            )
  
  return (history, tpu_model.get_weights())


def earlyStoppingCallback(delta, patience):
  return tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=delta, patience=patience, verbose=0, mode='min')

def tensorboardCallback(logDir, saveGraph = False, saveImages = False):
  return tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=0, write_graph=saveGraph, write_images=saveImages, update_freq = 'epoch')

def checkpointCallback(fileName, bestOnly = True, weightsOnly = True):
  return tf.keras.callbacks.ModelCheckpoint(filepath = fileName,monitor='val_loss',verbose=0,save_best_only=bestOnly, save_weights_only=weightsOnly, mode='min')
  
def buildTimestampName(localtime):
  return str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min) + '_' + str(localtime.tm_sec)

def cleanCheckpointCallbackWeights(path):
  #elimino tutti i pesi salvati che hanno un validation loss inferiore al migliore
  savedWeights = list(filter(lambda fn:fn.endswith(".hdf5"),os.listdir(path)))
  assert len(savedWeights) > 0, "Non sono stati salvati dei pesi"
  maxEpoch = max([int(fn.split("_")[-3]) for fn in savedWeights])
  for fn in savedWeights:
    if int(fn.split("_")[-3]) != maxEpoch:
      os.remove(path + fn)

def reduceLearningRaterAfterNIteration(reduceAfter):
  def learningRateFunc(learningRate, iteration):
    if iteration < reduceAfter:
      return learningRate
    else:
      return learningRate * pow(2,-(iteration-reduceAfter))
  return learningRateFunc

def reduceLearningRate():
  return reduceLearningRaterAfterNIteration(0)