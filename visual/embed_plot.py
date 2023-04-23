import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure



def embedPlot(window, xLocation, yLocation, size, xData, yData):
    fig = Figure(figsize = (size, size), dpi = 100, facecolor = "black", edgecolor = "blue")
    plot = fig.add_subplot(111, facecolor = "black")

    # plot.set_xlim(left = 0, right = 10000)
    # plot.set_ylim(bottom = 0, top = 10)

    plot.spines["right"].set_color("blue")
    plot.spines["bottom"].set_color("blue")
    plot.spines["left"].set_color("blue")
    plot.spines["top"].set_color("blue")

    plot.tick_params(color = "blue", labelcolor = "blue")
    #plot.xaxis.label.set_color("blue")

    plot.plot(xData, yData, ".", color = "lime")
    plotCanvas = FigureCanvasTkAgg(fig, master = window)
    plotCanvas.draw()
    plotCanvas.get_tk_widget().place(x = xLocation, y = yLocation)
    return (plot, plotCanvas)