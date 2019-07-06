import tensorflow as tf
import numpy as np

class Models:
  def __init__(self):
    self.__models = {}
    self.__modelsTrainingInfo = {}
  
  def addModel(self, modelName, model, trainingInfo):
    if modelName in self.__models:
      raise ValueError('Modello già presente o ridefinizione di modello')
    self.__models[modelName] = model
    self.__modelsTrainingInfo[modelName] = trainingInfo
  
  def getModelsName(self):
    return [name for name in self.__models.keys()]

  def getModelTrainingInfo(self, modelName):
    if modelName not in self.__models:
      raise ValueError("Modello", modelName, "non presente")
    return self.__modelsTrainingInfo[modelName]
  def getModel(self, modelName):
    if modelName not in self.__models:
      raise ValueError("Modello", modelName, "non presente")
    return self.__models[modelName]

def buildDenseLayer(inputLayer, layers, size, regularizers = 0.0, useBatch = True, flattenInput = False, leaky = False,dropout = 0.0, name =''):
  outputLayer = inputLayer
  oldName = name
  inputNameNumber = int(inputLayer.name.split("_")[0].replace("L",""))
  if oldName == '':
    name = 'L' + str(inputNameNumber+1)
  else:
    name = oldName
  if flattenInput == True:
    outputLayer = tf.keras.layers.Flatten(name = name + "_FLATTEN")(outputLayer)
  for i in range(layers):
    if regularizers > 0.0:
      outputLayer = tf.keras.layers.Dense(size, kernel_regularizer = tf.keras.regularizers.l2(l=regularizers), name= name+"_DENSE" + str(i))(outputLayer)
    else:
      outputLayer = tf.keras.layers.Dense(size, name = name +"_DENSE" + str(i))(outputLayer)
    if useBatch == True:
      outputLayer = tf.keras.layers.BatchNormalization(name = name +"_BATCH" + str(i))(outputLayer)
    if leaky == False:
      outputLayer =tf.keras.layers.ReLU(name = name +"_RELU" + str(i))(outputLayer)
    else:
      outputLayer = tf.keras.layers.LeakyReLU(name = name +"_LEAKY" + str(i))(outputLayer)
    if dropout > 0.0:
      randSeed = np.random.randint(np.iinfo(np.int32).max)
      outputLayer = tf.keras.layers.Dropout(dropout, seed=randSeed, name = name +"_DROPOUT" + str(i))(outputLayer)
  return outputLayer

def buildTransConvLayer(inputLayer, layers, size, kernelSize, strides = 1, flatten = False, useBatchNorm = True, leaky = False,withoutActivation = False, padding = "same",name=''):
  outputLayer = inputLayer
  for i in range(layers):
    outputLayer = tf.keras.layers.Conv2DTranspose(filters=size, kernel_size = kernelSize,strides = strides,padding = padding, name=name + str(i))(outputLayer)
    if useBatchNorm == True:
        outputLayer = tf.keras.layers.BatchNormalization()(outputLayer)
    if withoutActivation == False:
        if leaky == False:
          outputLayer =tf.keras.layers.ReLU()(outputLayer)
        else:
          outputLayer = tf.keras.layers.LeakyReLU()(outputLayer)
  if flatten == True:
    outputLayer = tf.keras.layers.Flatten()(outputLayer)
  return outputLayer


def buildConvLayer(inputLayer, layers, size, kernelSize, flatten = False, useBatchNorm = True, leaky = False,withoutActivation = False, padding = "same",name=''):
  outputLayer = inputLayer
  oldName = name
  inputNameNumber = int(inputLayer.name.split("_")[0].replace("L",""))
  if oldName == '':
    name = 'L' + str(inputNameNumber+1)
  else:
    name = oldName
  #print(inputLayer.name)
  #print(oldName)
  for i in range(layers):
    outputLayer = tf.keras.layers.Conv2D(filters=size, kernel_size = kernelSize, padding = padding, name=name + "_CONV" + str(i))(outputLayer)
    if useBatchNorm == True:
        outputLayer = tf.keras.layers.BatchNormalization(name = name+ "_BATCH" + str(i))(outputLayer)
    if withoutActivation == False:
        if leaky == False:
          outputLayer =tf.keras.layers.ReLU(name = name+ "_RELU" + str(i))(outputLayer)
        else:
          outputLayer = tf.keras.layers.LeakyReLU(name = name+ "_LEAKY" + str(i))(outputLayer)
  if flatten == True:
    outputLayer = tf.keras.layers.Flatten(name = name+ "_FLATTEN")(outputLayer)
  return outputLayer

def buildLocConvLayer(inputLayer, layers, size, kernelSize, flatten = False, useBatchNorm = True, leaky = False,withoutActivation = False, padding = "same"):
  outputLayer = inputLayer
  for i in range(layers):
    outputLayer = tf.keras.layers.LocallyConnected2D(filters=size, kernel_size = kernelSize, padding = padding)(outputLayer)
    if useBatchNorm == True:
        outputLayer = tf.keras.layers.BatchNormalization()(outputLayer)
    if withoutActivation == False:
        if leaky == False:
          outputLayer =tf.keras.layers.ReLU()(outputLayer)
        else:
          outputLayer = tf.keras.layers.LeakyReLU()(outputLayer)
  if flatten == True:
    outputLayer = tf.keras.layers.Flatten()(outputLayer)
  return outputLayer

def buildConvBlock(inputLayer, layers, size, kernelSize, poolSize = 2, poolStrides = 2, flatten = False, useBatch = True, name = ''):
  inputNameNumber = int(inputLayer.name.split("_")[0].replace("L",""))
  if name == '':
    name = "L"+str(inputNameNumber+1)

  outputLayer = buildConvLayer(inputLayer, layers, size, kernelSize, flatten = False, useBatchNorm = useBatch, leaky = False,withoutActivation = False, padding = "same", name = name +"_CONV")
  outputLayer = tf.keras.layers.MaxPool2D(pool_size = poolSize, strides = poolStrides,  padding='same', name = name + "_MAXPOOL")(outputLayer)
  if flatten == True:
    outputLayer = tf.keras.layers.Flatten(name = name + "_FLATTEN")(outputLayer)
  return outputLayer

def buildMaxPoolLayer(inputLayer, poolSize = 2, poolStrides = 2, flattenOutput = False, padding='same', name =''):
  inputNameNumber = int(inputLayer.name.split("_")[0].replace("L",""))
  if name == '':
    name = "L" + str(inputNameNumber+1) + "_MAXPOOL"
  outputLayer = tf.keras.layers.MaxPool2D(pool_size = poolSize, strides = poolStrides,  padding=padding, name = name)(inputLayer)
  if flattenOutput == True:
    outputLayer = tf.keras.layers.Flatten(name = "L" + str(inputNameNumber+1) + "_FLATTEN")(outputLayer)
  return outputLayer
    

def buildInceptionBlock(inputLayer, size1x1 = 64, size3x3 = 64, size5x5 = 64, sizeMaxPool = 64):
  b1 = buildConvLayer(inputLayer, layers= 1, size = size1x1, kernelSize = 1)

  b2 = buildConvLayer(inputLayer, layers= 1, size = size3x3, kernelSize = 1)
  b2 = buildConvLayer(b2, layers= 1, size = size3x3, kernelSize = 3)

  b3 = buildConvLayer(inputLayer, layers= 1, size = size5x5, kernelSize = 1)
  b3 = buildConvLayer(b3, layers= 1, size = size5x5, kernelSize = 5)
  
  b4 = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 1,  padding='same')(inputLayer)
  b4 = buildConvLayer(b4, layers= 1, size = sizeMaxPool, kernelSize = 1)

  return tf.keras.layers.concatenate([b1,b2,b3,b4])

def buildResidualInceptionBlock(inputLayer, size1x1 = 64, size3x3 = 64, size5x5 = 64, sizeMaxPool = 64):
  b1 = buildConvLayer(inputLayer, layers= 1, size = size1x1, kernelSize = 1)

  b2 = buildConvLayer(inputLayer, layers= 1, size = size3x3, kernelSize = 1)
  b2 = buildConvLayer(b2, layers= 1, size = size3x3, kernelSize = 3)

  b3 = buildConvLayer(inputLayer, layers= 1, size = size5x5, kernelSize = 1)
  b3 = buildConvLayer(b3, layers= 1, size = size5x5, kernelSize = 5)
  
  b4 = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 1,  padding='same')(inputLayer)
  b4 = buildConvLayer(b4, layers= 1, size = sizeMaxPool, kernelSize = 1)

  return tf.keras.layers.concatenate([b1,b2,b3,b4,inputLayer])

def buildResidualInceptionBlockV2(inputLayer, size1x1 = 64, size3x3 = 64, size5x5 = 64, sizeMaxPool = 64, dimensionalityReduction = -1):
  b1 = buildConvLayer(inputLayer, layers= 1, size = size1x1, kernelSize = 1)

  b2 = buildConvLayer(inputLayer, layers= 1, size = size3x3, kernelSize = 1)
  b2 = buildConvLayer(b2, layers= 1, size = size3x3, kernelSize = 3)

  b3 = buildConvLayer(inputLayer, layers= 1, size = size5x5, kernelSize = 1)
  b3 = buildConvLayer(b3, layers= 1, size = size5x5, kernelSize = 5)
  
  b4 = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 1,  padding='same')(inputLayer)
  b4 = buildConvLayer(b4, layers= 1, size = sizeMaxPool, kernelSize = 1)

  outputLayer = tf.keras.layers.concatenate([b1,b2,b3,b4,inputLayer])
  if dimensionalityReduction > 0:
    outputLayer = buildConvLayer(outputLayer, layers= 1, size = dimensionalityReduction, kernelSize = 1)
  return outputLayer

def buildResidualInceptionBlockV3(inputLayer, size1x1 = 64, size3x3 = 64, size5x5 = 64, sizeMaxPool = 64, dimensionalityReduction = -1):
  b1 = buildConvLayer(inputLayer, layers= 1, size = size1x1, kernelSize = 1)

  b2 = buildConvLayer(inputLayer, layers= 1, size = size3x3, kernelSize = 1)
  b2 = buildConvLayer(b2, layers= 1, size = size3x3, kernelSize = 3)

  b3 = buildConvLayer(inputLayer, layers= 1, size = size5x5, kernelSize = 1)
  b3 = buildConvLayer(b3, layers= 1, size = size5x5, kernelSize = 3)
  b3 = buildConvLayer(b3, layers= 1, size = size5x5, kernelSize = 3)
  
  b4 = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 1,  padding='same')(inputLayer)
  b4 = buildConvLayer(b4, layers= 1, size = sizeMaxPool, kernelSize = 1)

  outputLayer = tf.keras.layers.concatenate([b1,b2,b3,b4,inputLayer])
  if dimensionalityReduction > 0:
    outputLayer = buildConvLayer(outputLayer, layers= 1, size = dimensionalityReduction, kernelSize = 1)
  return tf.keras.layers.BatchNormalization()(outputLayer)

def buildDenseSoftmax(inputLayer, numberOfLabels = 43, flattenInput = False, name =""):
  outputLayer = inputLayer
  inputNameNumber = int(inputLayer.name.split("_")[0].replace("L",""))
  if flattenInput == True:
    outputLayer = tf.keras.layers.Flatten(name = "L"+str(inputNameNumber+1)+"_FLATTEN")(outputLayer)
  outputLayer = tf.keras.layers.Dense(numberOfLabels, name = "L"+str(inputNameNumber+1)+ "_DENSE")(outputLayer)
  outputLayer = tf.keras.layers.BatchNormalization(name = "L"+str(inputNameNumber+1)+"_BATCH")(outputLayer)
  return tf.keras.layers.Softmax(name = "L"+str(inputNameNumber+1)+"_SOFTMAX")(outputLayer)

# se dense size > 0 viene aggiunto un layer dense con dimensione indicata e poi altri 43 neuroni per l'output
def buildGlobalSoftmax(inputLayer, numberOfLabels = 43, reductionLayerSize = 0, kernelSize = 3, denseDepth = 0, denseSize = 0, dropout = 0.0):
  inputNameNumber = int(inputLayer.name.split("_")[0].replace("L",""))
  if denseSize > 0 and denseDepth <= 0:
    raise ValueError("depth <= 0 and size > 0")
  outputLayer = inputLayer
  if reductionLayerSize > 0:
    outputLayer = buildConvLayer(outputLayer, layers = 1, size = reductionLayerSize, kernelSize = kernelSize, name = "L"+str(inputNameNumber+1)+"_GLOBAL_REDUCTION")
  outputLayer = tf.keras.layers.GlobalAveragePooling2D(name = "L"+str(inputNameNumber+1)+"_GLOBAL_POOLING")(outputLayer)
  if denseDepth > 0:
    outputLayer = buildDenseLayer(outputLayer, denseDepth, denseSize, regularizers = 0.01, flattenInput = True, leaky = False,dropout = dropout, name = "L"+str(inputNameNumber+1)+"_GLOBAL_DENSE")
    outputLayer = buildDenseSoftmax(outputLayer, numberOfLabels = numberOfLabels, flattenInput = False)
  else:
    outputLayer = tf.keras.layers.Softmax(name = "L"+str(inputNameNumber+1)+"_GLOBAL_SOFTMAX")(outputLayer)
  return outputLayer

def buildResidualInceptionBlockV4(inputLayer, residualLayer, size1x1 = 64, size3x3 = 64, size5x5 = 64, sizeMaxPool = 64, dimensionalityReduction = -1, preserveResidual = True):
  inputNameNumber = int(inputLayer.name.split("_")[0].replace("L",""))
  b1 = buildConvLayer(inputLayer, layers= 1, size = size1x1, kernelSize = 1, name ="L"+str(inputNameNumber+1)+"_INC_B1X1")

  b2 = buildConvLayer(inputLayer, layers= 1, size = size3x3, kernelSize = 1, name="L"+str(inputNameNumber+1)+"_INC_B3X3_1")
  b2 = buildConvLayer(b2, layers= 1, size = size3x3, kernelSize = 3, name="L"+str(inputNameNumber+1)+"_INC_B3X3_2")

  b3 = buildConvLayer(inputLayer, layers= 1, size = size5x5, kernelSize = 1, name="L"+str(inputNameNumber+1)+"_INC_B5X5_1")
  b3 = buildConvLayer(b3, layers= 1, size = size5x5, kernelSize = 5, name="L"+str(inputNameNumber+1)+"_INC_B5X5_2")
  
  b4 = buildMaxPoolLayer(inputLayer, poolSize = 3, poolStrides = 1, flattenOutput = False, padding='same', name="L"+str(inputNameNumber+1)+"_INC_BMAX_1")
  b4 = buildConvLayer(b4, layers= 1, size = sizeMaxPool, kernelSize = 1, name ="L"+str(inputNameNumber+1)+"_INC_BMAX_2")

  if dimensionalityReduction > 0:
    if preserveResidual == True:
      inceptionLayer = tf.keras.layers.concatenate([b1,b2,b3,b4], name ="L"+str(inputNameNumber+1)+"_INC_CONC_TOWER")
      dimReductionLayer = buildConvLayer(inceptionLayer, layers= 1, size = dimensionalityReduction, kernelSize = 1, name ="L"+str(inputNameNumber+1)+"_INC_REDUCTION")
      outputLayer = tf.keras.layers.concatenate([dimReductionLayer,residualLayer], name ="L"+str(inputNameNumber+1)+"_INC_CONC_REDUCTIONwithRESIDUAL")
    else:
      outputLayer = tf.keras.layers.concatenate([b1,b2,b3,b4, residualLayer], name ="L"+str(inputNameNumber+1)+"_INC_CONC_TOWERwithRESIDUAL")
      outputLayer = buildConvLayer(outputLayer, layers= 1, size = dimensionalityReduction, kernelSize = 1, name ="L"+str(inputNameNumber+1)+"_INC_REDUCTION")
  else:
    outputLayer = tf.keras.layers.concatenate([b1,b2,b3,b4, residualLayer],name ="L"+str(inputNameNumber+1)+"_INC_CONC_TOWERwithRESIDUAL")
  return tf.keras.layers.BatchNormalization(name ="L"+str(inputNameNumber+1)+"_INC_BATCH")(outputLayer)

def buildResNetBlock(inputLayer, size1, size2, size3):
  # https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
  inputNameNumber = int(inputLayer.name.split("_")[0].replace("L",""))
  outputLayer = buildConvLayer(inputLayer, 1, size1, 1,flatten=False, padding = "valid", name="L"+str(inputNameNumber+1)+"_RES1")
  outputLayer = buildConvLayer(outputLayer, 1, size2, 3,flatten=False, name="L"+str(inputNameNumber+1)+"_RES2")
  outputLayer = buildConvLayer(outputLayer, 1, size3, 1, withoutActivation = True, flatten=False, padding="valid", name="L"+str(inputNameNumber+1)+"_RES3")
  shortcut = buildConvLayer(inputLayer, 1, size3, 1, withoutActivation=True,padding="valid",  name="L"+str(inputNameNumber+1)+"_RES_SHORTCUT")
  addLayer = tf.keras.layers.Add(name="L"+str(inputNameNumber+1)+"_RES_ADD" )([outputLayer, shortcut])
  return tf.keras.layers.ReLU(name="L"+str(inputNameNumber+1)+"_RES_RELU")(addLayer)