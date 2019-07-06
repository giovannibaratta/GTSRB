import tensorflow as tf
import numpy as np
import CommonUtils as utils
from colorama import Fore, Style


def testModel(tpu_address, modelsBuilder, weights, testData):
  tfVersion = utils.tfVersion()
  if tfVersion["MAJOR"] > 1:
    raise NotImplementedError("Training per la versione 2.0 non implementata")
  else:
    if tfVersion["MINOR"] > 13:
      return __testModel114(tpu_address, modelsBuilder, weights, testData)
    else:
      return __testModel113(tpu_address, modelsBuilder, weights, testData)

def __testModel114(tpu_address, modelsBuilder, weights, testData):
  predictions = []
  for modelIndex in range(len(modelsBuilder)):
    tf.reset_default_graph() 
    
    resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)
    tf.contrib.distribute.initialize_tpu_system(resolver)
    strategy = tf.contrib.distribute.TPUStrategy(resolver)
    with strategy.scope():
      model = modelsBuilder[modelIndex]()

      model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), #fake LR
          loss=tf.keras.losses.sparse_categorical_crossentropy,
          metrics=['sparse_categorical_accuracy'])
      
      for weightIndex in range(len(weights[modelIndex])):
        print('Test model', modelIndex, 'weights', weightIndex)
        model.load_weights(weights[modelIndex][weightIndex])
        predictions.append(model.predict(testData, batch_size = 512, verbose = 1))
  return predictions

def __testModel113(tpu_address, modelsBuilder, weights, testData):
    predictions = []
    for modelIndex in range(len(modelsBuilder)):
      tf.reset_default_graph() 
      tpu_model_loaded = tf.contrib.tpu.keras_to_tpu_model(
                          modelsBuilder[modelIndex](),
                          strategy=tf.contrib.tpu.TPUDistributionStrategy(
                          tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)))

      tpu_model_loaded.compile(
              optimizer=tf.train.AdamOptimizer(learning_rate = 0.001), #fake LR
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['sparse_categorical_accuracy']
          )
          
      for weightIndex in range(len(weights[modelIndex])):
        print('Test model', modelIndex, 'weights', weightIndex)
        tpu_model_loaded.load_weights(weights[modelIndex][weightIndex])
        predictions.append(tpu_model_loaded.predict(testData))
    return predictions

def predictionEvaluation(predictions, labels, modelsToUse, topN = 5, weights = None):
    # per ogni top (top1, top2, ..., topN) contiene un dict che tiene traccia delle inferenze corrette
    topCorrectDistribution = []
    # per ogni top (top1, top2, ..., topN) contiene un dict che tiene traccia delle inferenze sbagliate
    topWrongDistribution =  []
    numberOfClasses = max(labels) + 1
    #init stat
    for topIndex in range(topN):
        topCorrectDistribution.append({})
        topWrongDistribution.append({})
        for classIndex in range(numberOfClasses):
            topCorrectDistribution[topIndex][classIndex] = 0
            topWrongDistribution[topIndex][classIndex] = 0

    for predictionIndex in range(len(labels)):
        prediction = np.zeros((numberOfClasses), dtype="float")
        # preparo il vettore inferenza
        for modelIndex in modelsToUse:
            if weights != None:
                #altero la predict di ogni classe con un peso 
                prediction = np.add(prediction, (predictions[modelIndex][predictionIndex]  * weights[modelIndex])/sum(predictions[modelIndex][predictionIndex]  * weights[modelIndex]))
            else:
                prediction = np.add(prediction, predictions[modelIndex][predictionIndex])
        # verifico se l'inferenza è corretta o sbagliata
        for topIndex in range(topN):
            hotPredictions = np.argmax(prediction)  
            if hotPredictions == labels[predictionIndex]:
                #inferenza corretta, aggiorno le statistiche correct dei top successivi a partire da quello attuale
                for topUpdate in range(topIndex, topN):
                    topCorrectDistribution[topUpdate][hotPredictions] = topCorrectDistribution[topUpdate][hotPredictions] + 1
                break
            # inferenza sbagliata
            topWrongDistribution[topIndex][labels[predictionIndex]] = topWrongDistribution[topIndex][labels[predictionIndex]] + 1 
            prediction[hotPredictions] = 0

    for topIndex in range(topN):
        correct = sum(topCorrectDistribution[topIndex].values())
        wrong = sum(topWrongDistribution[topIndex].values())
        print("TOP",topIndex+1, "corrette", correct, "su", (correct+wrong), "(",wrong,")" ,float(correct)/(wrong + correct)*100, "%")
    print("")

    print("TOP1 distribuzione:")
    
    print(topCorrectDistribution[0])

    for key, value in sorted(topCorrectDistribution[0].items(), key=lambda keyValuePair: keyValuePair[0]):
        correct = value
        wrong = topWrongDistribution[0][key]
        print(key, correct, wrong)
        percentage = float(correct)/(wrong + correct)*100
        if percentage >= 99.0:
            color  = Fore.GREEN
        if percentage > 95.0 and percentage < 99.0:
            color = Fore.YELLOW
        if percentage <= 95.0:
            color = Fore.RED
        print(Style.RESET_ALL, color, "Classe ", Style.BRIGHT, key, Style.RESET_ALL, color, "] corrette ", correct, " su ", correct + wrong," (",wrong,") ",Style.BRIGHT, percentage, Style.RESET_ALL, color, "%", sep="")