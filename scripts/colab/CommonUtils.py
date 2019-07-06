import numpy as np
import random as rn
import matplotlib.pyplot as plt
import os
import tensorflow as tf


def tfVersion():
  version = tf.__version__.split(".")
  major = int(version[0])
  minor = int(version[1])
  patch = version[2].split("-")
  return {
    "MAJOR" : major, "MINOR" : minor, "PATCH" : patch[0]
  }

def resetSeed(offset = 0):
  np.random.seed(4390183 + offset)
  rn.seed(5790283 + offset)
  if tfVersion()["MAJOR"] < 2:
    tf.set_random_seed(9832795 + offset)
  else:
    tf.random.set_seed(9832795 + offset)
  os.environ['PYTHONHASHSEED'] = '0'

  
def saveLossPlot(figPath,figTitle, data, frequency = 1, showFigure = False):
  fig, ax = plt.subplots()
  
  maxX = 0
  minY = 99999
  maxY = 0

  for curve, label in data:
    yAxis = curve
    xAxis = np.arange(1, len(curve)+1, 1) * frequency
    ax.plot(xAxis, yAxis, label = str(label))
    maxX = max(maxX, len(yAxis) + 1)
    minY = min(minY, min(curve))
    maxY = max(maxY, max(curve))

  ax.set(xlabel='epochs', ylabel='loss',
         title=figTitle + ' loss')
  ax.grid()

  plt.xticks(np.arange(1, maxX*frequency, max(2.0,round(maxX*frequency/15))))
  plt.ylim(bottom=0)
  plt.legend()
  if showFigure == True:
    plt.show()
  fig.savefig(figPath)
  plt.close()
  
def saveAccuracyPlot(figPath,figTitle, data, frequency = 1, showFigure = False):
  fig, ax = plt.subplots()
  
  maxX = 0
  minY = 99999
  maxY = 0

  for curve, label in data:
    yAxis = curve
    xAxis = np.arange(1, len(curve)+1, 1) * frequency
    ax.plot(xAxis, yAxis, label = str(label))
    maxX = max(maxX, len(yAxis) + 1)
    minY = min(minY, min(curve))
    maxY = max(maxY, max(curve))

  ax.set(xlabel='epochs', ylabel='accuracy',
         title=figTitle + ' accuracy')
  ax.grid()


  plt.xticks(np.arange(1, maxX*frequency, max(2.0,round(maxX*frequency/15))))
  plt.locator_params(axis='y', nbins=20)
  plt.ylim(top=1.0)
  plt.legend()
  if showFigure == True:
    plt.show()
  fig.savefig(figPath)
  plt.close()

def generateRandomImage(generator, image):
  randSeed = np.random.randint(np.iinfo(np.int32).max)
  transOp = generator.get_random_transform((), seed=randSeed)
  del transOp['flip_horizontal']
  del transOp['flip_vertical']
  transOp['zy'] = transOp['zx']
  return generator.apply_transform(image,transform_parameters = transOp)

def makeDirs(dirList):
  for d in dirList:
    os.mkdir(d)
    
def generateIndices(num, end):
    if num < end:
        indices = np.random.choice(end, num, replace = False)
    else:
        indices = np.zeros((num), dtype = np.int16)
        count = 0
        while num > 0:
            newIndices = np.random.choice(end, min(num, end), replace = False)
            for index in newIndices:
                indices[count] = index
                count += 1
            num = num - end
    return indices