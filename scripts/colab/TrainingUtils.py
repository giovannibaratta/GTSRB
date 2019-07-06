import os
import tensorflow as tf
import shutil
import CommonUtils as utils
import numpy as np
import time
from colorama import Fore, Style
import pickle
# trace di errori
import traceback

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
    self.__validationFrequency = 1
    self.__validationFrequencyFT = 1
    self.__classWeights = None

    
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
      validationFrequency = None,
      validationFrequencyFT = None,
      classWeights = None
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
      if self.__freezeFrom > self.__fineTuningIterations:
        raise ValueError("Il numero di iterazioni è",self.__fineTuningIterations)
      self.__freezeFrom = freezeFrom

    if learningRateUpdater != None:
      self.__learningRateUpdater = learningRateUpdater
      
    if lossFunction != None:
      self.__lossFunction = lossFunction
    if metrics != None:
      self.__metrics = metrics
    if validationFrequency != None:
      self.__validationFrequency = validationFrequency
    if validationFrequencyFT != None:
      self.__validationFrequencyFT = validationFrequencyFT
    if classWeights != None:
      self.__classWeights = classWeights  
      
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
  def getValidationFrequency(self):
    return self.__validationFrequency
  def getClassWeights(self):
    return self.__classWeights
  def getValidationFrequencyFT(self):
    return self.__validationFrequencyFT  
  
def convertModelToTPU(model, tpu_address):
  return tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
    tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address))
  )


def retrieveWeights(modelBuilder, weightsPath, tpu_address):
  tfVersion = utils.tfVersion()
  if tfVersion["MAJOR"] > 1:
    return __retrieveWeights200(modelBuilder, weightsPath, tpu_address)
  else:
    if tfVersion["MINOR"] > 13:
      return __retrieveWeights114(modelBuilder, weightsPath, tpu_address)
    else:
      return __retrieveWeights113(modelBuilder, weightsPath, tpu_address)

def __retrieveWeights200(modelBuilder, weightsPath, tpu_address):
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu=tpu_address)
  tf.config.experimental_connect_to_host(cluster_resolver.master())
  tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
  strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
  with strategy.scope():
    model = modelBuilder()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01 ),
        lossFunction=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['sparse_categorical_accuracy'])
    weightsFile = os.listdir(weightsPath)[0]
    model.load_weights(weightsPath + weightsFile)
    weights = model.get_weights()
  return weights

def __retrieveWeights114(modelBuilder, weightsPath, tpu_address):
  resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)
  tf.contrib.distribute.initialize_tpu_system(resolver)
  strategy = tf.contrib.distribute.TPUStrategy(resolver)
  with strategy.scope():
    model = modelBuilder()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01 ),
        lossFunction=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['sparse_categorical_accuracy'])
    weightsFile = os.listdir(weightsPath)[0]
    model.load_weights(weightsPath + weightsFile)
    weights = model.get_weights()
  return weights

def __retrieveWeights113(modelBuilder, weightsPath, tpu_address):
  tpu_model = convertModelToTPU(modelBuilder(), tpu_address)
  weightsFile = os.listdir(weightsPath)[0]
  tpu_model.load_weights(weightsPath + weightsFile)
  return tpu_model.get_weights()

def __trainModel200(tpu_address, modelBuilder, learningRate, trainingData, trainingLabels, validationData, validationLabels,
              epochs, shuffle, batchSize, lossFunction , metrics, callbacks = [], weights = None, weightsFromFile = None, verbose = 1,
              validationFrequency = 1, classWeights = None):
  # trasformo il modello da tipo keras a modello compatibile con le TPU
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu=tpu_address)
  tf.config.experimental_connect_to_host(cluster_resolver.master())
  tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
  strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
  
  with strategy.scope():
    model = modelBuilder()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate ),
        loss=lossFunction,
        metrics=metrics)

    if weights != None:
      model.set_weights(weights)
    else:
      if weightsFromFile != None:
       model.load_weights(weightsFromFile,  by_name =True)
    history = model.fit(
                x = trainingData,
                y = trainingLabels,
                batch_size = batchSize,
                validation_data = (validationData, validationLabels),
                epochs = epochs,
                shuffle = shuffle,
                verbose = verbose,
                callbacks=callbacks,
                class_weight = classWeights
              )
  
  return (history, model.get_weights())

def trainModel(tpu_address, modelBuilder, learningRate, trainingData, trainingLabels, validationData, validationLabels,
              epochs, shuffle, batchSize, lossFunction , metrics, callbacks = [], weights = None, weightsFromFile = None, verbose = 1,
              validationFrequency = 1, classWeights = None):
  tfVersion = utils.tfVersion()
  if tfVersion["MAJOR"] > 1:
    return __trainModel200(tpu_address, modelBuilder, learningRate, trainingData, trainingLabels, validationData, validationLabels,
              epochs, shuffle, batchSize, lossFunction , metrics, callbacks , weights , weightsFromFile , verbose, validationFrequency, classWeights )
  else:
    if tfVersion["MINOR"] > 13:
      return __trainModel114(tpu_address, modelBuilder, learningRate, trainingData, trainingLabels, validationData, validationLabels,
              epochs, shuffle, batchSize, lossFunction , metrics, callbacks , weights , weightsFromFile , verbose, validationFrequency, classWeights )
    else:
      return __trainModel113(tpu_address, modelBuilder, learningRate, trainingData, trainingLabels, validationData, validationLabels,
              epochs, shuffle, batchSize, lossFunction , metrics, callbacks , weights , weightsFromFile , verbose, validationFrequency, classWeights )

def __trainModel114(tpu_address, modelBuilder, learningRate, trainingData, trainingLabels, validationData, validationLabels,
              epochs, shuffle, batchSize, lossFunction , metrics, callbacks = [], weights = None, weightsFromFile = None, verbose = 1,
              validationFrequency = 1, classWeights = None):
  # trasformo il modello da tipo keras a modello compatibile con le TPU
  resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)
  tf.contrib.distribute.initialize_tpu_system(resolver)
  strategy = tf.contrib.distribute.TPUStrategy(resolver)
  with strategy.scope():
    model = modelBuilder()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
        loss=lossFunction,
        metrics=metrics)

    if weights != None:
      model.set_weights(weights)
    else:
      if weightsFromFile != None:
       model.load_weights(weightsFromFile,  by_name =True)

    steps = len(trainingData) // batchSize
    
    history = model.fit(
                x = trainingData,
                y = trainingLabels,
                batch_size = batchSize,
                validation_data = (validationData, validationLabels),
                epochs = epochs,
                shuffle = shuffle,
                verbose = verbose,
                callbacks= callbacks,
                steps_per_epoch = steps,
                validation_freq=validationFrequency
              )
  
  return (history, model.get_weights())

def __trainModel113(tpu_address, modelBuilder, learningRate, trainingData, trainingLabels, validationData, validationLabels,
              epochs, shuffle, batchSize, lossFunction , metrics, callbacks = [], weights = None, weightsFromFile = None, verbose = 1,
              validationFrequency = 1, classWeights = None):
  # trasformo il modello da tipo keras a modello compatibile con le TPU
  tpu_model = convertModelToTPU(modelBuilder(), tpu_address)

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
              callbacks=callbacks,
              class_weight = classWeights
            )
  
  return (history, tpu_model.get_weights())


def earlyStoppingCallback(delta, patience):
  return tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=delta, patience=patience, verbose=0, mode='min')

def tensorboardCallback(logDir, saveGraph = False, saveImages = False):
  return tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=0, write_graph=saveGraph, write_images=saveImages, update_freq = 'epoch')

def checkpointCallback(fileName, bestOnly = True, weightsOnly = True, period = 1):
  return tf.keras.callbacks.ModelCheckpoint(filepath = fileName,monitor='val_loss',verbose=0,save_best_only=bestOnly, save_weights_only=weightsOnly, mode='min', period = period)

def progbarCallback():
  return tf.keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)

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


def train(tpu_address,
  savePath,
  models, 
  verboseTraining = 1,
  mainTrainingEarlyStoppingDelta = 0.005,
  mainTrainingEarlyStoppinPatience = 10,
  fineTuningEarlyStoppingDelta = 0.001,
  fineTuningEarlyStoppinPatience = 10,
  stringToLog = None,
  lastSession = []):

  tfVersion = utils.tfVersion()
  if tpu_address != '':
    useTPU = True

  for modelName in models.getModelsName():
    # timestamp per identificare univocamente una run
    localtime = time.localtime(time.time())
    # recupero il modello e i suoi parametri per il training
    model = models.getModel(modelName)
    trainingInfo = models.getModelTrainingInfo(modelName)
    lossFunction = trainingInfo.getLossFunction()
    metrics = trainingInfo.getMetrics()
    resumeTraining = trainingInfo.getResumeTraining()
    tData = trainingInfo.getTrainingData()
    tLabels = trainingInfo.getTrainingLabels()
    valData = trainingInfo.getValidationData()
    valLabels = trainingInfo.getValidationLabels()
    validationFrequency = trainingInfo.getValidationFrequency()
    validationFrequencyFT = trainingInfo.getValidationFrequencyFT()
    classWeights = trainingInfo.getClassWeights()
    weights = trainingInfo.getWeights()
    weightsFromFile = trainingInfo.getWeightsFromFile()
    mainEpochs = trainingInfo.getMainEpochs()
    shuffle = trainingInfo.getShuffle()
    batchSize = trainingInfo.getBatchSize()
    fineTuniningEpochs = trainingInfo.getFineTuningEpochs()

    #Prepare la directory principale training/modelNameAndtime/
    skipModelDefinition = False
    learningListToResume = []
    if resumeTraining != None:
      modelPath = savePath + modelName + "_" + resumeTraining[0] + "/"
      skipModelDefinition = True
      learningListToResume = [info[0] for info in resumeTraining[1]]
    else:
      modelPath = savePath + modelName + "_" + buildTimestampName(localtime) + "/"
    tensorboardPath = modelPath + "tensorboard/"
    try:
      lastSession.append(modelPath)
      
      if skipModelDefinition == False:
        os.mkdir(modelPath)
        
        with open(modelPath + 'model.json', 'w') as modelFile:
          jsonModel = model().to_json()
          modelFile.write(jsonModel)
        trainingParametersFile = "trainingParameters.txt"
      else:
        trainingParametersFile = "resumedTrainingParameters.txt"

      with open(modelPath + trainingParametersFile,"w") as trainingLog:
        if stringToLog != None:
          trainingLog.write(stringToLog)
        freezeLayer = trainingInfo.getFreezeLayer()
        trainingLog.write("freezeLayer : " + str(freezeLayer)+ "\n")
        if freezeLayer == True:
          trainingLog.write("freezeFrom : " + str(trainingInfo.getFreezeFrom())+ "\n")
          trainingLog.write("freezeIndex : " + str(trainingInfo.getLayersToFreeze())+ "\n")

      # per ogni learning rate eseguto un ciclo di training
      for learningRate in trainingInfo.getLearningRateList():
        # non funziona in v2
        if useTPU == True and tfVersion['MAJOR'] < 2 :
          # se non si resetta la prima iterazione impiega sempre più tempo per il training
          tf.reset_default_graph() 
        
        print(Style.BRIGHT, "\n" + modelName + "] : main training con lr" + str(learningRate), Style.RESET_ALL)
        learningPath = modelPath + str(learningRate) + "/" #training/modelNameAndTime/learningRate/
        mainTrainingPath = learningPath + "Main/" #training/modelNameAndTime/learningRate/Main/
        weightsPath = mainTrainingPath + "weights/" #training/modelNameAndTime/learningRate/Main/weights/
          
        if learningRate not in learningListToResume:
          # Preparo le direcotry 
          plotsPath = mainTrainingPath + "plots/" #training/modelNameAndTime/learningRate/Main/plots/
          historyPath = mainTrainingPath + "history/" #training/modelNameAndTime/learningRate/Main/history/
          utils.makeDirs([learningPath, mainTrainingPath, weightsPath, plotsPath, historyPath])
          # main training
          tensorboardIterationPath = tensorboardPath + "MainTraining_" + str(learningRate) + "/"
          checkpointPath = '/tmp/epoch_{epoch:02d}_valLoss_{val_loss:.4f}.hdf5'
          callbacks = [
              tensorboardCallback(logDir = tensorboardIterationPath,saveGraph = True, saveImages = True),
              earlyStoppingCallback(mainTrainingEarlyStoppingDelta , mainTrainingEarlyStoppinPatience ),
              checkpointCallback(fileName = checkpointPath, bestOnly = True, weightsOnly = True, period = validationFrequency)
          ]
          
          history, weights = trainModel(
              tpu_address, model, learningRate,
              tData, tLabels,
              valData, valLabels,
              mainEpochs, shuffle, batchSize,
              lossFunction, metrics,
              callbacks = callbacks,
              weights = weights,
              weightsFromFile = weightsFromFile,
              verbose = verboseTraining,
              validationFrequency = validationFrequency,
              classWeights = classWeights)
          
          cleanCheckpointCallbackWeights('/tmp/')
          lastWeights = list(filter(lambda fn: fn.startswith('epoch') and fn.endswith('.hdf5'),os.listdir('/tmp/')))[0]
          # copio i pesi migliori nella cartella di gdrive
          shutil.move('/tmp/' + lastWeights, weightsPath + lastWeights)
          # salvo i grafici
          utils.saveLossPlot(plotsPath + "validationLoss.png", "Validation",[(history.history['val_loss'], learningRate)], frequency = validationFrequency)
          utils.saveLossPlot(plotsPath + "traininLoss.png", "Training",[(history.history['loss'], learningRate)])
          if 'val_sparse_categorical_accuracy' in history.history:
            utils.saveAccuracyPlot(plotsPath + "validationAccuracy.png", "Validation",[(history.history['val_sparse_categorical_accuracy'],learningRate)], frequency = validationFrequency)
          if 'sparse_categorical_accuracy' in history.history:
            utils.saveAccuracyPlot(plotsPath + "trainingAccuracy.png", "Training",[(history.history['sparse_categorical_accuracy'],learningRate)])
          with open(historyPath + "history.history","wb") as pickleOut:
            pickle.dump(history.history, pickleOut)
        else:
          # se devo riprendere un training assumo che il main ci sia già stato e riprendo da li
          weights = retrieveWeights(model, weightsPath, tpu_address) 
        # fine-tuning
        if trainingInfo.getFineTuningIterations() > 0:
          fineTuningPath = learningPath + "FineTuning/" #training/modelNameAndTime/learningRate/FineTuning/
          utils.makeDirs([fineTuningPath])
          if learningRate not in learningListToResume:
            learningRateUpdater = trainingInfo.getLearningRateUpdater()
          else:
            learningRateUpdater = list(filter(lambda lrAndUpdater:lrAndUpdater[0] == learningRate,resumeTraining[1]))[0][1]
          for iteration in range(trainingInfo.getFineTuningIterations()):
            
            if useTPU == True and tfVersion['MAJOR'] < 2:
              # se non si resetta la prima iterazione impiega sempre più tempo
              tf.reset_default_graph() 

            fineTuningLR = learningRateUpdater(learningRate, iteration + 1)
            print(Style.BRIGHT, "\n" + modelName + "] : fine tuning con lr" + str(fineTuningLR) + "partendo da" + str(learningRate), Style.RESET_ALL)
            #preparo le directory
            fineTuningLRPath = fineTuningPath + str(fineTuningLR) + "/" #training/modelNameAndTime/learningRate/FineTuning/LearningRate/
            weightsPath = fineTuningLRPath + "weights/" #training/modelNameAndTime/learningRate/FineTuning/LearningRate/weights/
            plotsPath = fineTuningLRPath + "plots/" #training/modelNameAndTime/learningRate/FineTuning/LearningRate/plots/
            historyPath = fineTuningLRPath + "history/" #training/modelNameAndTime/learningRate/FineTuning/LearningRate/history/
            utils.makeDirs([fineTuningLRPath, weightsPath, plotsPath, historyPath])
            #freezo i layer se necessario
            if trainingInfo.getFreezeLayer() == True and trainingInfo.getFreezeFrom() - 1 >= iteration and trainingInfo.getLayersToFreeze >= 0:
              for layerIndex in range(trainingInfo.getLayersToFreeze + 1):
                model.layers[layerIndex].trainable = False
            #faccio il training
            tensorboardIterationPath = tensorboardPath + "FineTuning_" + str(learningRate) +  "_" + str(fineTuningLR) + "/"
            checkpointPath = '/tmp/epoch_{epoch:02d}_valLoss_{val_loss:.4f}.hdf5'
            callbacks = [
                tensorboardCallback(logDir = tensorboardIterationPath,saveGraph = True, saveImages = True),
                earlyStoppingCallback(fineTuningEarlyStoppingDelta , fineTuningEarlyStoppinPatience ),
                checkpointCallback(fileName = checkpointPath, bestOnly = True, weightsOnly = True, period = validationFrequencyFT)
            ]
            
            history, weights = trainModel(
                tpu_address, model, fineTuningLR,
                tData, tLabels,
                valData, valLabels,
                fineTuniningEpochs, shuffle, batchSize,
                lossFunction, metrics,
                callbacks = callbacks, weights = weights, verbose = verboseTraining,
                validationFrequency = validationFrequencyFT,
                classWeights = classWeights)
            cleanCheckpointCallbackWeights('/tmp/')
            lastWeights = list(filter(lambda fn: fn.startswith('epoch') and fn.endswith('.hdf5'),os.listdir('/tmp/')))[0]
            shutil.move('/tmp/' + lastWeights, weightsPath + lastWeights)
            utils.saveLossPlot(plotsPath + "validationLoss.png", "Validation",[(history.history['val_loss'], learningRate)],frequency = validationFrequencyFT)
            utils.saveLossPlot(plotsPath + "traininLoss.png", "Training",[(history.history['loss'], learningRate)])
            if 'val_sparse_categorical_accuracy' in history.history:
              utils.saveAccuracyPlot(plotsPath + "validationAccuracy.png", "Validation",[(history.history['val_sparse_categorical_accuracy'],learningRate)],frequency = validationFrequencyFT)
            if 'sparse_categorical_accuracy' in history.history:
              utils.saveAccuracyPlot(plotsPath + "trainingAccuracy.png", "Training",[(history.history['sparse_categorical_accuracy'],learningRate)])
            with open(historyPath + "history.history","wb") as pickleOut:
              pickle.dump(history.history, pickleOut)
        # fine ciclo sui learnin rate
        return history
    except Exception as e:
      print("Errore durante il training.\n")
      lastSession.pop()
      traceback.print_exc()
      if 'resumeTraining' in locals():
        if resumeTraining != None:
          print("Cleaning", modelPath)
          shutil.rmtree(modelPath)
      else:
        print("Cleaning", modelPath)
        shutil.rmtree(modelPath)