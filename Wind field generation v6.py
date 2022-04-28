# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:07:13 2021

@author: Oliver
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import timeit


baseWindSpeed = 10
windSpeedMultiplier = 0.7   # Max gust size


shape = []

for a in range(2, 10):
    z = np.zeros((a*2-1,a*2-1))

    ci,cj=a-1,a-1
    cr=a-1

    I,J=np.meshgrid(np.arange(z.shape[0]),np.arange(z.shape[1]))
    dist=np.sqrt((I-ci)**2+(J-cj)**2)
    z[np.where(dist<=cr)]=1

    shape.append(z)

shapes = dict(enumerate(shape))

class gustElement:
    def __init__(self, magnitude, position, radius, direction):
        self.magnitude = magnitude
        self.position = position
        self.radius = radius
        self.direction = direction/180 * math.pi
        self.directionVec = [-math.sin(self.direction), -math.cos(self.direction)]

    def updatePosition(self):
        self.position[0] -= self.magnitude * math.cos(self.direction)/baseWindSpeed
        self.position[1] -= self.magnitude * math.sin(self.direction)/baseWindSpeed


size = 50
resolution = 1
velocitySize = (size - 1) * resolution
variation = 30  # shift size

windDirectionDeg = 15

numGusts = int(size / 2)
gusts = [gustElement(baseWindSpeed * random.randrange(150, 200, 1)/100,
                     [random.randint(0, size - 1), random.randint(0, size - 1)],
                     random.randint(3, 6),
                     windDirectionDeg + random.randint(-variation, variation)) for a in range(numGusts)]

windDirectionRadCorrected = (windDirectionDeg + 180)/180 * math.pi  # 180 factor added to rotate arrow

xComDir = np.sin(windDirectionRadCorrected) * np.ones((size, size))
yComDir = np.cos(windDirectionRadCorrected) * np.ones((size, size))

windSpeedArray = baseWindSpeed * np.ones((velocitySize + 1, velocitySize + 1))
gustArray = np.zeros((velocitySize + 1, velocitySize + 1))


x = np.arange(0, size)
y = np.arange(0, size)

X1, Y1 = np.meshgrid(np.arange(0, (velocitySize + 1)/resolution, 1/resolution), np.arange(0, (velocitySize + 1)/resolution, 1/resolution))  # mesh for velocity plot
X2, Y2 = np.meshgrid(np.arange(0, size), np.arange(0, size))  # mesh for arrows


def addGustShape(matrix1, matrix2, xypos):
    x, y = xypos
    if matrix2.shape[0] == 1:
        matrix1[x, y] += matrix2
    else:
        h1, w1 = matrix1.shape
        h2, w2 = matrix2.shape

        x1min = max(0, x)
        y1min = max(0, y)
        x1max = max(min(x + w2, w1), 0)
        y1max = max(min(y + h2, h1), 0)

        x2min = max(0, -x)
        y2min = max(0, -y)
        x2max = min(-x + w1, w2)
        y2max = min(-y + h1, h2)

        matrix1[x1min:x1max, y1min:y1max] += matrix2[x2min:x2max, y2min:y2max]


def changeShiftShape(matrix1, matrix2, xypos, direction):
    x, y = xypos
    if matrix2.shape[0] == 1:
        if direction == "x":
            matrix2[matrix2 < 0.01] = np.sin(windDirectionRadCorrected)
            matrix1[x, y] = matrix2
        elif direction == "y":
            matrix2[matrix2 < 0.01] = np.cos(windDirectionRadCorrected)
            matrix1[x, y] = matrix2
    else:
        h1, w1 = matrix1.shape
        h2, w2 = matrix2.shape

        x1min = max(0, x)
        y1min = max(0, y)
        x1max = max(min(x + w2, w1), 0)
        y1max = max(min(y + h2, h1), 0)

        x2min = max(0, -x)
        y2min = max(0, -y)
        x2max = min(-x + w1, w2)
        y2max = min(-y + h1, h2)

        if direction == "x":
            matrix2[matrix2 < 0.01] = np.sin(windDirectionRadCorrected)
            matrix1[x1min:x1max, y1min:y1max] = matrix2[x2min:x2max, y2min:y2max]
        elif direction == "y":
            matrix2[matrix2 < 0.01] = np.cos(windDirectionRadCorrected)
            matrix1[x1min:x1max, y1min:y1max] = matrix2[x2min:x2max, y2min:y2max]


def refreshFrame(windSpeedArray, gusts):
    gustArray = np.zeros((size, size))
    xComDir = np.sin(windDirectionRadCorrected) * np.ones((size, size))
    yComDir = np.cos(windDirectionRadCorrected) * np.ones((size, size))

    for gust in gusts:  # create gustArray and adjust directions
        position = np.array([round(a) for a in gust.position])

        addGustShape(gustArray, gust.magnitude * shapes[gust.radius], position - (gust.radius - 1))

        changeShiftShape(xComDir, gust.directionVec[0] * shapes[gust.radius], position - (gust.radius - 1), direction="x")
        changeShiftShape(yComDir, gust.directionVec[1] * shapes[gust.radius], position - (gust.radius - 1), direction="y")

        gustArray[gustArray > baseWindSpeed * (windSpeedMultiplier + 1)] = baseWindSpeed * (windSpeedMultiplier + 1)

    superimposedWind = windSpeedArray + gustArray

    for gust in gusts:  # replace when they reach the edges
        gust.updatePosition()
        if gust.position[0] < 0 or gust.position[1] < 0 or gust.position[0] > size - 1 or gust.position[1] > size - 1:
            gusts.remove(gust)
            gusts.append(gustElement(random.randint(round(0.25 * baseWindSpeed),
                                    windSpeedMultiplier * baseWindSpeed),
                                    [size - 1, random.randint(0, size - 1)],
                                    random.randint(1, 3),
                                    windDirectionDeg + random.randint(-variation, variation)))
    return superimposedWind

def timeFunc():
    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped

    wrapped = wrapper(refreshFrame, windSpeedArray, gusts)

    print(timeit.timeit(wrapped, number=10))
    print(timeit.timeit(wrapped, number=100))


y = []

for i in range(1000):
    a = refreshFrame(windSpeedArray, gusts)[25][25]
    if a > 20:
        y.append(20)
    else:
        y.append(a)

x = [i for i in range(1000)]

plt.plot(x, y)
plt.show()