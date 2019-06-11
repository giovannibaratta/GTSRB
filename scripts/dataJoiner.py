'''
Raggruppa le singole immagini in un unico file per label
'''
import sys
import numpy as np
import os
import time
import random
import math
import cv2
from multiprocessing import Process

# Restituisce una matrice contenente le immagini.
# Le immagini vengono suddivise per gruppo, il prefisso del nome indica il gruppo.
# Ad esempio 0000_0000.ppm è la prima immagine del gruppo 0, 0004_0029.ppm è
# l'ultima immagine del gruppo 4. Ogni gruppo contiene 30 immagini
def groupAndResizeImages(dirPath, targetWidth, targetHeight):
    ppmFiles = list(filter(lambda fileName: fileName.endswith('.ppm'), os.listdir(dirPath)))
    images = []
    for fileName in ppmFiles:
        with open(dirPath + "\\" + fileName, 'rb') as ppmFile:
            imageIndex = int(fileName.split("_")[1].replace(".ppm",""))
            if imageIndex == 0:
                group = []
            group.append(resizeImage(convertImage(ppmFile.read()), targetWidth, targetHeight))
            if imageIndex == 29:
                images.append(group)
    return images

# conversione da ppm a matrice di byte
def convertImage(data):
    #salto la prima linea perchè è il formato
    currentIndex = 3
    if data[currentIndex] == 35:
        #salto la linea perchè è un commento
        while data[currentIndex] != 10:
            currentIndex += 1
        currentIndex += 1

    #lettura width
    widthByte = []
    heightByte = []

    while data[currentIndex] != 32:
        widthByte.append(chr(data[currentIndex]))
        currentIndex = currentIndex + 1

    width = int(''.join(widthByte))

    #lettura height
    currentIndex = currentIndex + 1
    while data[currentIndex] != 10:
        heightByte.append(chr(data[currentIndex]))
        currentIndex = currentIndex + 1

    height = int(''.join(heightByte))

    currentIndex = currentIndex + 5
    data = data[currentIndex:len(data)+1]
    currentIndex = 0
    image = np.zeros((height,width,3), dtype="uint8")

    for pixelIndex in range(len(data)):
        rowIndex =  (pixelIndex // 3) // width
        colIndex = (pixelIndex // 3) % width
        depth = pixelIndex % 3
        image[rowIndex][colIndex][depth] = data[pixelIndex]
    return image

def resizeImage(image, targetWidth, targetHeight):
    height = len(image)
    width= len(image[0])
    top, left, right, bottom = 0,0,0,0
    newIm = image
    
    if width != height:
        #pad image
        diff = width - height 
        if diff > 0:
            #pad sopra e sotto
            if diff % 2 == 0:
                top = diff // 2
                bottom = diff // 2
            else:
                top = math.ceil(diff//2)
                bottom = diff - top
        else:
            #pas sinistra e destra
            diff = -diff
            if diff % 2 == 0:
                left = diff // 2
                right = diff // 2
            else:
                left = math.ceil(diff//2)
                right = diff - left
        newIm = cv2.copyMakeBorder(newIm,top,bottom,left,right,cv2.BORDER_REFLECT)
    return cv2.resize(newIm, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)

def handleClass(dirPrefix, dirName, targetWidth, targetHeight,trainingDir, validationDir):
    classLabel = int(dirName)
    images = groupAndResizeImages(dirPrefix + dirName, targetWidth, targetHeight)
    #splitting 
    trainingImages = []
    validationImages = []
    for groupId in range(len(images)):
        # il 30 % dei dati viene usato come validazione
        imagesIndex = np.random.choice(10, 3, replace = False)
        for index in range(10):
            if index in imagesIndex:
                validationImages.append(images[groupId][index])
            else:
                trainingImages.append(images[groupId][index])
        imagesIndex = np.random.choice(10, 3, replace = False) + 10
        for index in range(10,20):
            if index in imagesIndex:
                validationImages.append(images[groupId][index])
            else:
                trainingImages.append(images[groupId][index])
        imagesIndex = np.random.choice(10, 3, replace = False) + 20
        for index in range(10,20):
            if index in imagesIndex:
                validationImages.append(images[groupId][index])
            else:
                trainingImages.append(images[groupId][index])
    #saving
    saveImages(trainingImages, classLabel, trainingDir, targetWidth, targetHeight)
    saveImages(validationImages, classLabel, validationDir, targetWidth, targetHeight)
    print(classLabel, str(len(images) * 30), len(trainingImages), len(validationImages))

def saveImages(images, classLabel, outputPath, targetWidth, targetHeight):
    imgMap = np.memmap(outputPath + 'resized' + str(classLabel), dtype=np.uint8,
                mode='w+', shape=(len(images),targetWidth,targetHeight,3))

    labelMap = np.memmap(outputPath + 'labels' + str(classLabel), dtype=np.uint8,
                mode='w+', shape=(len(images)))

    copiedImages = 0
    for image in images:
        imgMap[copiedImages] = image
        labelMap[copiedImages] = classLabel
        copiedImages += 1
        
    del labelMap
    del imgMap
    
    with open(outputPath + 'num' + str(classLabel),'w') as out:
        out.write(str(len(images)))

# percorso che contiene tutte le cartelle delle immagini separate per classi (0000,0001,0002,...)
dirPrefix = sys.argv[1]
# cartella in cui salvare le immagini per il training
trainingDir = sys.argv[2]
# cartella in cui salvare le immagine per la validazione
validationDir = sys.argv[3]
# larghezza target delle nuove immagini
targetWidth = int(sys.argv[4])
# altezza target delle nuove immagini
targetHeight = int(sys.argv[5])

np.random.seed(4612383)

if __name__ == '__main__':
    print("")
    print('INPUT DIR', dirPrefix)
    print('TRAINING DIR', trainingDir)
    print('VALIDATION DIR', validationDir)

    directories = os.listdir(dirPrefix)
    processList = []
    for dirName in directories:
        if len(processList) > 3:
            processList.pop(0).join()
        process = Process(target = handleClass, args=(dirPrefix, dirName, targetWidth, targetHeight, trainingDir, validationDir))
        processList.append(process)
        process.start()

    for process in processList:
        process.join()

    print('Termino')