"""
	This file contains SMO (Sequencial Minimal Optimization) for training SVM
	(Support Vector Machine)
"""

import numpy as np

DEFAULT_TOL = 0.02


class SupportVectorMachine(object):

    """This class describe a support vector machine (SVM)"""

    def __init__(self, datas, C, tol=DEFAULT_TOL):
        self.C = C
        self.tol = tol

        self.X = datas[:, :-1]
        self.Y = datas[:, -1]

        self.alphas_or_b_changed = True

        # Number of training examples and features
        (self.m, self.n) = datas.shape

        # Create list of lagrange multipliers _alphas, and initialise it to 0
        self._alphas = np.zeros(self.m)

        # Initialise threshold _b to 0
        self._b = 0

        # Initialise hyperplane normal vector _w to 0
        self._w = np.zeros(self.n)

        self.errors = np.array([self.h(x) - y for (x, y) in
                                zip(self.X, self.Y)])

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
            self._w = self.X.T.dot(self.Y * self._alphas)
            self.alphas_or_b_changed = False

        return self._w

    w = property(_get_w)
    alphas = property(_get_alphas, _set_alphas)
    b = property(_get_b, _set_b)

    def _examine_example(self, i2):
        x2 = self.X[i2]
        y2 = self.Y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.h(x2) - y2

        r2 = y2 * E2

        # What about the case r == 0 and (alpha == 0 or alpha == C) ?
        if r2 < -self.tol and alpha2 < self.C or r2 > self.tol and alpha2 > 0:
            # Try to optimise non bounded values first
            # It means that at least an other on exists
            if (self.alphas == 0).sum() + (self.alphas == self.C).sum() >= 2:
                # First, begin by second choice heuristic
                max_error_diff = 0
                i1 = None
                for position, error_diff in \
                        enumerate(abs(self.errors - E2)):

                    if error_diff > max_error_diff:
                        max_error_diff = error_diff
                        i1 = position

                if self.take_step(i1, i2):
                    return True

                # If second choice heuristic is not successful, loop over all
                # non-zero and non-C alphas
                for i1, value in enumerate(self.alphas):
                    if value not in (0, self.C):
                        if self.take_step(i1, i2):
                            return True

                # If no non-zero and non-C alphas, loop over all alphas
                for i1, value in enumerate(self.alphas):
                    if self.take_step(i1, i2):
                        return True
        return False

    def take_step(self, i2, i1):
        if i2 == i1:
            return False

        x1 = self.X[i1]
        y1 = self.Y[i1]
        alpha1 = self.alphas[i1]
        E1 = self.h(x1) - y1

        x2 = self.X[i2]
        y2 = self.Y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.h(x2) - y2

        s = y1 * y2

        # Compute L and H
        if s != 1:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 + alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return False

        # Compute needed kernels
        # For the moment, only linear kernel is taken into account
        k11 = x1.dot(x1)
        k12 = x1.dot(x2)
        k22 = x2.dot(x2)

        # Compute eta
        eta = 2 * k12 - k11 - k22

        # Compute alpha2_new
        # For the moment the case eta = 0 (means that x1 = x2) is not taken
        # into account. (It will throw an exception)
        alpha2_new = alpha2 - y2 * (E1 - E2) / eta
        if alpha2_new < L:
            alpha2_new = L
        elif alpha2_new > H:
            alpha2_new = H

        if alpha2_new < 1e-8:
            alpha2_new = 0
        elif alpha2_new > self.C - 1e-8:
            alpha2_new = self.C

        # Compute alpha1_new
        alpha1_new = alpha1 * s * (alpha2_new - alpha2)

        # Compute b_new
        b1 = E1 + y1 * (alpha1_new - alpha1) * k11 + \
            y2 * (alpha2_new - alpha2) * k12 + self.b

        b2 = E2 + y1 * (alpha1_new - alpha1) * k12 + \
            y2 * (alpha2_new - alpha2) * k22 + self.b

        if alpha1_new not in (0, self.C) and alpha2_new not in (0, self.C):
            b_new = b1  # = b2
        else:
            print b1, b2
            b_new = (b1 + b2) / 2

        self.b = b_new

        # Store alpha1_new and alpha2_new
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        return True

    def h(self, vector):
        """Compute the hypothesis for the vector 'vector'"""
        return self.w.dot(vector) + self.b

    def train(self):
        """Train the support vector machine"""

        # TODO : write comment
        num_changed = 0

        # TODO : write comment
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0

            if examine_all:
                for i in range(self.m):
                    num_changed += self._examine_example(i)
            else:
                for i, value in enumerate(self.alphas):
                    if value not in (0, self.C):
                        num_changed += self._examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
