# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:20:05 2021

@author: Oliver
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

baseWindSpeed = 10
windSpeedMultiplier = 2   # Max gust size

shapes = {1: np.array([1]),
          2: np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]),
          3: np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]])}


class gustElement:
    def __init__(self, magnitude, position, radius):
        self.magnitude = magnitude
        self.position = position
        self.radius = radius

    def updatePosition(self):
        self.position[0] -= self.magnitude/baseWindSpeed


size = 20
resolution = 1
velocitySize = (size - 1) * resolution
variation = 10

windDirectionDeg = 0

numGusts = int(size / 2)
gusts = [gustElement(random.randint(baseWindSpeed, windSpeedMultiplier * baseWindSpeed),
                     [random.randint(0, size - 1), random.randint(0, size - 1)], random.randint(1, 3)) for a in range(numGusts)]

windDirectionRadCorrected = (windDirectionDeg + 180)/180 * math.pi  # 180 factor added to rotate arrow

windSpeedArray = baseWindSpeed * np.ones((velocitySize + 1, velocitySize + 1))
gustArray = np.zeros((velocitySize + 1, velocitySize + 1))

# %% Showing the starting wind field
x = np.arange(0, size)
y = np.arange(0, size)

X1, Y1 = np.meshgrid(np.arange(0, (velocitySize + 1)/resolution, 1/resolution), np.arange(0, (velocitySize + 1)/resolution, 1/resolution))  # mesh for velocity plot
X2, Y2 = np.meshgrid(np.arange(0, size), np.arange(0, size))  # mesh for arrows

fig = plt.figure(figsize=(13, 13))

plt.contourf(X1, Y1, windSpeedArray, cmap="winter")
plt.colorbar()
plt.clim(vmin=baseWindSpeed, vmax=(windSpeedMultiplier + 1) * baseWindSpeed)
plt.quiver(X2, Y2, math.sin(windDirectionRadCorrected), math.cos(windDirectionRadCorrected), pivot="mid")

ax = fig.gca()
ax.set_xticks(np.arange(-0.5, size + 0.5))
ax.set_yticks(np.arange(-0.5, size + 0.5))  # makes figure fit on the page

plt.axis("off")
# %%


def addGustShape(matrix1, matrix2, xypos):
    x, y = xypos
    if matrix2.shape[0] == 1:
        matrix1[x, y] += matrix2
    else:
        h1, w1 = matrix1.shape
        h2, w2 = matrix2.shape
    
        # get slice ranges for matrix1
        x1min = max(0, x)
        y1min = max(0, y)
        x1max = max(min(x + w2, w1), 0)
        y1max = max(min(y + h2, h1), 0)
    
        # get slice ranges for matrix2
        x2min = max(0, -x)
        y2min = max(0, -y)
        x2max = min(-x + w1, w2)
        y2max = min(-y + h1, h2)
    
        matrix1[x1min:x1max, y1min:y1max] += matrix2[x2min:x2max, y2min:y2max]


def refreshFrame(windSpeedArray, gusts):
    gustArray = np.zeros((size, size))
    directionArray = np.zeros((size, size))
    for gust in gusts:  # create gustArray and adjust directions
        position = np.array([round(a) for a in gust.position])
        addGustShape(gustArray, gust.magnitude * shapes[gust.radius], position - (gust.radius - 1))
        gustArray[gustArray > baseWindSpeed * (windSpeedMultiplier + 1)] = baseWindSpeed * (windSpeedMultiplier + 1)
        

    superimposedWind = windSpeedArray + gustArray

    fig = plt.figure(figsize=[13, 13])

    plt.contourf(X1, Y1, superimposedWind, cmap="winter")
    plt.colorbar()
    plt.clim(vmin=baseWindSpeed, vmax=(windSpeedMultiplier + 1) * baseWindSpeed)
    plt.quiver(X2, Y2, math.sin(windDirectionRadCorrected), math.cos(windDirectionRadCorrected), pivot="mid")

    ax = fig.gca()
    ax.set_xticks(np.arange(-0.5, size + 0.5))
    ax.set_yticks(np.arange(-0.5, size + 0.5))  # makes figure fit on the page

    plt.axis("off")

    for gust in gusts:  # replace when they reach the end
        gust.updatePosition()
        if gust.position[0] < 0 or gust.position[1] < 0:
            gusts.remove(gust)
            gusts.append(gustElement(random.randint(baseWindSpeed,
                                    windSpeedMultiplier * baseWindSpeed),
                                    [size - 1, random.randint(0, size - 1)],
                                    random.randint(1, 3)))


while True:
    refreshFrame(windSpeedArray, gusts)
    plt.pause(0.1)
