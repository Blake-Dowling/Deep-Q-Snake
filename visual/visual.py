import time
import tkinter
from tkinter import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import visual.embed_plot as embed_plot
import math

####################Adding Plots####################
class MatrixPlot:
    def __init__(self, window, x, y):
        (self.plot, self.plotCanvas) = embed_plot.embedPlot(window, x, y, 2, [], [] )
    # (plot1, plotCanvas1) = embed_plot.embedPlot(window, 200, 300, 2, [], [] )
    # (plot2, plotCanvas2) = embed_plot.embedPlot(window, 0, 300, 2, [], [] )
    def plotMatrix(self, matrix):
        (numRows, numCols) = np.shape(matrix)
        inputX = []
        inputY = []
        points = []
        created = []
        for row in range(numRows):
            for col in range(numCols):
                inputX.append(col)
                inputY.append(row)
                points.append(matrix[row][col])
        points = np.array(points)
        color = np.sqrt((points**2))/np.sqrt(2.0)
        rgb = plt.get_cmap('jet')(color)
        #inputX.reverse()
        inputY.reverse()
        newPoint = self.plot.scatter(inputX, inputY, color=rgb)
        plt.show()
        created.append(newPoint)
        self.plotCanvas.draw()
        #canvas.update()
        for obj in created:
            obj.remove()
    def plotKerasLayer(self, layer):
        w = layer.get_weights()[0]
        print("Weights: ", w)
        self.plotMatrix(w)

