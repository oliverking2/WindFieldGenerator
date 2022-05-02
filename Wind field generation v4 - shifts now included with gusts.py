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
windSpeedMultiplier = 1.7   # Max gust size


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
        self.visible = True
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
gusts = [gustElement(baseWindSpeed * round(random.normalvariate(0.75, 0.1),2),
                     [random.randint(0, size - 1), random.randint(0, size - 1)],
                     random.randint(1, 3),
                     windDirectionDeg + random.randint(-variation, variation)) for a in range(numGusts)]

gusts.sort(key=lambda x: x.magnitude)
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
    overlap = np.zeros((size, size))
    xComDir = np.sin(windDirectionRadCorrected) * np.ones((size, size))
    yComDir = np.cos(windDirectionRadCorrected) * np.ones((size, size))

    for gust in gusts:  # create gustArray and adjust directions
        position = np.array([round(a) for a in gust.position])

        addGustShape(gustArray, gust.magnitude * shapes[gust.radius], position - (gust.radius - 1))
        addGustShape(overlap, shapes[gust.radius], position - (gust.radius - 1))

        locsx, locsy = np.where(overlap>1)

        for i in range(len(locsx)):
            gustArray[locsx[i]][locsy[i]] = gust.magnitude

        overlap[overlap > 1] = 1
        changeShiftShape(xComDir, gust.directionVec[0] * shapes[gust.radius], position - (gust.radius - 1), direction="x")
        changeShiftShape(yComDir, gust.directionVec[1] * shapes[gust.radius], position - (gust.radius - 1), direction="y")

    # gustArray[gustArray > baseWindSpeed * (windSpeedMultiplier + 1)] = baseWindSpeed * (windSpeedMultiplier + 1)
    gusts.sort(key=lambda x: x.magnitude)

    superimposedWind = windSpeedArray + gustArray

    # fig = plt.figure(figsize=[13, 13])
    #
    # plt.contourf(X1, Y1, superimposedWind, cmap="winter")
    # plt.colorbar()
    # plt.clim(vmin=10, vmax=40)
    # plt.quiver(X2, Y2, xComDir, yComDir, pivot="mid")
    #
    # ax = fig.gca()
    # ax.set_xticks(np.arange(-0.5, size + 0.5))
    # ax.set_yticks(np.arange(-0.5, size + 0.5))  # makes figure fit on the page
    #
    # plt.axis("off")

    for gust in gusts:  # replace when they reach the edges
        gust.updatePosition()
        if gust.position[0] < 0 or gust.position[1] < 0 or gust.position[0] > size - 1 or gust.position[1] > size - 1:
            gusts.remove(gust)
            gusts.append(gustElement(baseWindSpeed * round(random.normalvariate(0.75, 0.1),2),
                                    [size - 1, random.randint(0, size - 1)],
                                    random.randint(1, 5),
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

#
# while True:
#     refreshFrame(windSpeedArray, gusts)
#     plt.pause(0.1)


y = []

length = 10000

for a in range(5):
    for i in range(length):
        b = refreshFrame(windSpeedArray, gusts)
        a = b[25][25]
        if a > 20:
            y.append(20)
        else:
            y.append(a)

    x = [i for i in range(length)]

    # plt.plot(x, y)
    # plt.show()

    x2, y2 = [], []
    for i in np.arange(15.0, 20.0, 0.1):
        a = round(i, 1)
        x2.append(a)
        y2.append(100 * y.count(a)/len(y))

    for a in [6, 9, 33, 42]:
        y2[a] = (y2[a-1] + y2[a+1])/2

# plt.bar(x2,y2,width=0.1)
# plt.show()

fig, ax = plt.subplots()
p1 = ax.bar(x2, y2, width=0.1)
ax.set_xlabel('Wind Speed (kts)')
ax.set_ylabel('Percentage (%)')
ax.set_title('A graph showing the distribution of Wind Speeds produced\n by the Wind Model')
fig.show()