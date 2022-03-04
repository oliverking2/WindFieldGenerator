# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:01:41 2021

@author: Oliver
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time

global count

size = 20
resolution = 1
velocitySize = (size - 1) * resolution
count = 0

windDirectionDeg = 0  # + np.random.randint(0, 30)
windDirectionRadCorrected = (windDirectionDeg + 180)/180 * math.pi  # 180 factor added to rotate arrow

minWindVelocity = 10
maxWindVelocity = 15

windVelocityArray = np.random.randint(minWindVelocity, maxWindVelocity, (velocitySize + 1, velocitySize + 1))  # randomise velocity field

x = np.arange(0, size)
y = np.arange(0, size)

X1, Y1 = np.meshgrid(np.arange(0, (velocitySize + 1)/resolution, 1/resolution), np.arange(0, (velocitySize + 1)/resolution, 1/resolution))  # mesh for velocity plot
X2, Y2 = np.meshgrid(np.arange(0, size), np.arange(0, size))  # mesh for arrows

fig = plt.figure(figsize=(13, 13))

plt.contourf(X1, Y1, windVelocityArray, cmap="winter")
plt.colorbar()
plt.quiver(X2, Y2, math.sin(windDirectionRadCorrected), math.cos(windDirectionRadCorrected), pivot="mid")

ax = fig.gca()
ax.set_xticks(np.arange(-0.5, size + 0.5))
ax.set_yticks(np.arange(-0.5, size + 0.5))  # makes figure fit on the page

plt.axis("off")


def randWindVel(minVel, maxVel):
    p = [(a/sum(range(minVel, maxVel + 1))) for a in range(minVel, maxVel + 1)][::-1]
    return np.random.choice([a for a in range(minVel, maxVel + 1)], p=p)


def newWindLine(prevLine, minVel, maxVel, size):
    return [randWindVel(minVel, maxVel) for a in range(0, size + 1)]


def updateGraph(windVelocityArray):
    if windDirectionDeg == 0:
        oldWindVelocityArray = windVelocityArray[1:]  # removes the last row of the matrix
        newLine = newWindLine(windVelocityArray[-1], minWindVelocity, maxWindVelocity, velocitySize)
        windVelocityArray = np.vstack((oldWindVelocityArray, newLine))

        fig = plt.figure(figsize=(13, 13))

        plt.contourf(X1, Y1, windVelocityArray, cmap="winter")
        plt.colorbar()
        plt.quiver(X2, Y2, math.sin(windDirectionRadCorrected), math.cos(windDirectionRadCorrected), pivot="mid")

        ax = fig.gca()
        ax.set_xticks(np.arange(-0.5, size + 0.5))
        ax.set_yticks(np.arange(-0.5, size + 0.5))  # makes figure fit on the page

        plt.axis("off")

        return windVelocityArray


while True:
    windVelocityArray = updateGraph(windVelocityArray)
    plt.pause(0.05)
    time.sleep(0.02)