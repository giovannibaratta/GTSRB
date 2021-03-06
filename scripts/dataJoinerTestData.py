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
def groupAndResizeImages(dirPrefix,fileList, targetWidth, targetHeight):
    images = []
    for fileName in fileList:
        with open(dirPrefix + "\\" + fileName, 'rb') as ppmFile:
            images.append(resizeImage(convertImage(ppmFile.read()), targetWidth, targetHeight))
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

def saveImages(images, outputPath, targetWidth, targetHeight):
    imgMap = np.memmap(outputPath + 'testData', dtype=np.uint8,
                mode='w+', shape=(len(images),targetWidth,targetHeight,3))
    copiedImages = 0
    for image in images:
        imgMap[copiedImages] = image
        copiedImages += 1
        
    del imgMap
    
    with open(outputPath + 'num','w') as out:
        out.write(str(len(images)))

inputDir = sys.argv[1]
outputDir = sys.argv[2]
targetWidth = int(sys.argv[3])
targetHeight = int(sys.argv[4])
labelFilePath = sys.argv[5]

if __name__ == '__main__':
    print("")
    print('INPUT DIR', inputDir)
    print('OUTPUT DIR', outputDir)
    
    files = list(filter(lambda fName : fName.endswith(".ppm"),os.listdir(inputDir)))
    files.sort(key = lambda fName: int(fName.replace(".ppm","")))
    images = groupAndResizeImages(inputDir, files, targetWidth, targetHeight)
    saveImages(images, outputDir, targetWidth, targetHeight)
    
    #genero le label
    with open(labelFilePath,"r") as labelFile:
        lines = labelFile.readlines()
    lines = lines[1:]
    labels = np.zeros((len(lines)),dtype="uint8")
    counter = 0
    for line in lines:
        labels[counter] = int(line.split(";")[7])
        counter += 1
    np.save(outputDir + "labels", labels)
    print('Termino')