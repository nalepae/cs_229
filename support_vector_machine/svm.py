"""
	This file contains SMO (Sequencial Minimal Optimization) for training SVM
	(Support Vector Machine)
"""

import numpy as np


class SupportVectorMachine(object):

    """This class describe a support vector machine (SVM)"""

    def __init__(self, datas):
        self.X = datas[:, :-1]
        self.y = datas[:, -1]

        # Number of training examples
        self.m = datas.shape[0]

        # Create list of lagrange multipliers alpha, and initialise it to 0
        self.alphas = np.zeros(self.m)

        # Initialise threshold to 0
        self.b = 0

    def w(self):
        """Compute normal hyperplane vector"""
        return self.X.T.dot(self.y * self.alphas)

    def h(self, vector):
        """Compute the hypothesis for the vector 'vector'"""
        return self.w().dot(vector) + self.b
