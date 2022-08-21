import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .calculations import centerOfMass
from .simulation import simulateStep

_animations = []


def plotObjectsStatic(objects):
    """Draw a matplotlib chart of our objects. `objects` argument is an Nx2 matrix of (x,y) coordinates"""
    objects = np.array(objects)
    plt.figure(figsize=(14, 8))
    # scatter the data, use max size for object size
    sizes = [min(100, x) for x in objects[:, 2]]
    plt.scatter(objects[:, 0], objects[:, 1], s=sizes, alpha=0.5)
    plt.show()


def plotObjects(objects, ax, lim=None):
    """Animate a matplotlib chart of our objects. `objects` argument is an Nx2 matrix of (x,y) coordinates"""
    # clear the frame
    ax.clear()

    # scatter the data, use max size for object size
    sizes = [min(100, x) for x in objects[:, 2]]
    ax.scatter(objects[:, 0], objects[:, 1], s=sizes, alpha=0.5)

    # plot the center of mass in red
    com = centerOfMass(objects)
    ax.scatter(com[0], com[1], s=25, c="red")

    if lim:
        ax.set_xlim(lim)
        ax.set_ylim(lim)


def animate(objects, time, delta_t, lim=None):
    frames = int(time / delta_t)
    fig, ax = plt.subplots(figsize=(14, 8))

    def animateStep(i):
        simulateStep(objects, delta_t)
        plotObjects(objects, ax, lim)

    ani = FuncAnimation(fig, animateStep, interval=100, frames=frames)
    _animations.append(ani)
    # writergif = PillowWriter(fps=30) 
    # ani.save("2.gif", writer=writergif)
    plt.show()
