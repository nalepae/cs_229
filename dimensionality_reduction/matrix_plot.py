"""This modules is useful to plot a matrix with arrows"""
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_X_MIN = -100
DEFAULT_X_MAX = 100
DEFAULT_X_NB = 20

DEFAULT_Y_MIN = -100
DEFAULT_Y_MAX = 100
DEFAULT_Y_NB = 20


class MatrixPlot(object):

    """Class useful to plot a matrix with arrows"""

    def __init__(self,
                 x_parameters=(DEFAULT_X_MIN, DEFAULT_X_MAX, DEFAULT_X_NB),
                 y_parameters=(DEFAULT_Y_MIN, DEFAULT_Y_MAX, DEFAULT_Y_NB)):

        x_min, x_max, self.x_nb = x_parameters
        y_min, y_max, self.y_nb = y_parameters

        x_lin = np.linspace(x_min, x_max, self.x_nb)
        y_lin = np.linspace(y_min, y_max, self.y_nb)

        self.xx, self.yy = np.meshgrid(x_lin, y_lin)

        self.lin_input = [self.xx.ravel(), self.yy.ravel()]

    def plot(self, matrix):
        """Plot the matrix"""
        uu, vv = np.dot(matrix, self.lin_input).reshape(2, self.x_nb,
                                                        self.y_nb)
        plt.grid()
        plt.quiver(self.xx, self.yy, uu, vv, scale_units='xy', angles='xy',
                   scale=1)
        plt.show()
