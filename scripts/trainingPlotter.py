import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pickle
import sys

modelsPath = sys.argv[1]

modelsToLoad = {
    "1D64" : (1e-4,"Modello 1"),
    "1D128" : (1e-4,"Modello 2"),
    "2D64" : (1e-4,"Modello 3"),
    "2D128" :(1e-4,"Modello 4"),
    "1C64_3" : (1e-4,"Modello 5"),
    "2C64_3" : (1e-4,"Modello 6"),
    "2C64_3_1D64" : (1e-4,"Modello 7"),
    "2C64_3_1MAX2_2_1D64" : (1e-4,"Modello 8"),
    "3C64_3_1I_G" : (1e-3,"Modello 9"),
}

'''
fig, ax = plt.subplots()
  
  maxX = 0
  minY = 99999
  maxY = 0

  for curve, label in data:
    yAxis = curve
    xAxis = np.arange(1, len(curve)+1, 1)
    ax.plot(xAxis, yAxis, label = str(label))
    maxX = max(maxX, len(yAxis) + 1)
    minY = min(minY, min(curve))
    maxY = max(maxY, max(curve))

  ax.set(xlabel='epochs', ylabel='loss',
         title=figTitle + ' loss')
  ax.grid()

  plt.xticks(np.arange(1, maxX, 2.0))
  plt.ylim(bottom=0)
  plt.legend()
  plt.show()
  fig.savefig(figPath)
'''
fig, ax = plt.subplots()

color =['#f21010','#b20505','#c14343','#751907','#1cef3c', '#68a028', '#74f286','#355b5a', '#e5e232']
plotCounter = 0
for key, value in modelsToLoad.items():
    with open(modelsPath + key + ".history","rb") as pickleIn:
        history = pickle.load(pickleIn)
    for training, learningRate in history:
        if learningRate != value[0]:
            continue
        yAxis = training['val_sparse_categorical_accuracy']
        xAxis = np.arange(1, len(yAxis)+1, 1)
        ax.plot(xAxis, yAxis, label = str(value[1]), color = color[plotCounter])
        ax.set(xlabel='epochs', ylabel='accuracy',
         title="Validation accuracy")
        break
    plotCounter += 1
plt.legend()
plt.ylim(bottom=0.7, top=1.0)

plt.grid()
fig.set_size_inches(16,9)
fig.savefig(modelsPath + "groupedAccuracy.png")
plt.show()
#2C64_3_1MAX2_2_1D64 1e-4, 2C64_3 1e-5 1e-4, 1D128 1e-4 1e-5, 3C64_3_1I_G 1e-3.

