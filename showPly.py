import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def showPly(ptWithColor,*args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract all coordinates and colors
    coordinates = ptWithColor[:, :3].reshape(-1, 3)
    colors = ptWithColor[:, 3:].reshape(-1, 3)

    # Plot all points at once
    ax.scatter(coordinates[:, 1], coordinates[:, 2], coordinates[:, 0], c=colors / 255.0, s=100)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(*args)

    plt.show()
