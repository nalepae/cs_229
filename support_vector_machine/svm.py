"""
	This file contains SMO (Sequencial Minimal Optimization) for training SVM
	(Support Vector Machine)
"""

import numpy as np


class SupportVectorMachine(object):

    """This class describe a support vector machine (SVM)"""

    def __init__(self, datas):
        self.alphas_or_b_changed = True

        self.X = datas[:, :-1]
        self.y = datas[:, -1]

        # Number of training examples and features
        (self.m, self.n) = datas.shape

        # Create list of lagrange multipliers _alphas, and initialise it to 0
        self._alphas = np.zeros(self.m)

        # Initialise threshold _b to 0
        self._b = 0

        # Initialise hyperplane normal vector _w to 0
        self._w = np.zeros(self.n)

    def _get_alphas(self):
        """Get _alphas"""
        return self._alphas

    def _set_alphas(self, alphas):
        """Set _alphas"""
        self.alphas_or_b_changed = True
        self._alphas = alphas

    def _get_b(self):
        """Get _b"""
        return self._b

    def _set_b(self, b):
        """Set _b"""
        self.alphas_or_b_changed = True
        self._b = b

    def _get_w(self):
        """Compute normal hyperplane vector"""
        if self.alphas_or_b_changed:
            self._w = self.X.T.dot(self.y * self._alphas)
            self.alphas_or_b_changed = False

        return self._w

    w = property(_get_w)
    alphas = property(_get_alphas, _set_alphas)
    b = property(_get_b, _set_b)

    def h(self, vector):
        """Compute the hypothesis for the vector 'vector'"""
        return self.w.dot(vector) + self.b
