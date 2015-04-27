"""
	This file contains SMO (Sequencial Minimal Optimization) for training SVM
	(Support Vector Machine)
"""

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_TOL = 1e-8
DEFAULT_MESH_X = 100
DEFAULT_MESH_Y = 100
DEFAULT_RELATIVE_MARGIN = 0.2
DEFAULT_SIGMA = 1


KERNEL_LINEAR = 0
KERNEL_GAUSSIAN = 1


class SupportVectorMachine(object):

    """This class describes a support vector machine (SVM)"""

    def __init__(self, datas, C, chosen_kernel=KERNEL_LINEAR,
                 sigma=DEFAULT_SIGMA, tol=DEFAULT_TOL):
        self.tol = tol
        self.chosen_kernel = chosen_kernel
        self.C = C
        self.sigma = sigma

        if self.chosen_kernel == KERNEL_LINEAR:
            self.X = datas[:, :-1]
            self.X_attributes = self.X
        elif self.chosen_kernel == KERNEL_GAUSSIAN:
            self.X_attributes = datas[:, :-1]
            m = self.X_attributes.shape[0]

            self.X = np.empty((m, m))
            for i in range(m):
                for j in range(m):
                    x1 = self.X_attributes[i]
                    x2 = self.X_attributes[j]
                    self.X[i, j] = self.gauss(x1, x2)

        self.Y = datas[:, -1]

        self.alphas_or_b_changed = True

        # Number of training examples and features
        (self.m, self.n) = self.X.shape

        # Create list of lagrange multipliers _alphas, and initialise it to 0
        self._alphas = np.zeros(self.m)

        # Initialise threshold _b to 0
        self._b = 0

        # Initialise hyperplane normal vector _w to 0
        self._w = np.zeros(self.n)

    def gauss(self, x1, x2):
        v = x1 - x2
        return np.exp(- np.dot(v, v) / (2 * self.sigma ** 2))

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

    def examine_example(self, i2):
        """Examine the training example i2 and optimisate it if needed"""
        x2 = self.X[i2]
        y2 = self.Y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.h(x2) - y2

        r2 = y2 * E2

        if r2 < - self.tol and alpha2 < self.C or r2 > self.tol and alpha2 > 0:
            for i1 in range(self.m):
                if self.take_step(i1, i2):
                    return True

        return False

    def take_step(self, i1, i2):
        """Take an optimisation step with training examples i1 and i2"""
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
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return False

        # Compute needed kernels
        # For the moment, only linear kernel is taken into account
        k11 = np.dot(x1, x1)
        k12 = np.dot(x1, x2)
        k22 = np.dot(x2, x2)

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

        if alpha2_new == alpha2:
            return False

        # Compute alpha1_new
        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)

        # Compute b_new
        b1 = E1 + y1 * (alpha1_new - alpha1) * k11 + \
            y2 * (alpha2_new - alpha2) * k12 + self.b

        b2 = E2 + y1 * (alpha1_new - alpha1) * k12 + \
            y2 * (alpha2_new - alpha2) * k22 + self.b

        if alpha1_new not in (0, self.C) and alpha2_new not in (0, self.C):
            b_new = b1  # = b2
        else:
            b_new = (b1 + b2) / 2

        self.b = b_new

        # Store alpha1_new and alpha2_new
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        return True

    def h(self, vector):
        """Compute the hypothesis for the vector 'vector'"""
        return np.dot(self.w, vector) - self.b

    def h_plot(self, vector):
        """Compute the hypothesis for the vector 'vector' in initial space"""
        if self.chosen_kernel == KERNEL_LINEAR:
            return self.h(vector)
        elif self.chosen_kernel == KERNEL_GAUSSIAN:
            vector_upspace = np.empty(self.m)
            for i in range(self.m):
                vector_upspace[i] = self.gauss(vector, self.X_attributes[i])

            return self.h(vector_upspace)

    def train(self):
        """Train the support vector machine"""
        one_change_at_least = True

        while one_change_at_least:
            one_change_at_least = False
            for i2 in range(self.m):
                if self.examine_example(i2):
                    one_change_at_least = True

    def plot(self, mesh_x=DEFAULT_MESH_X, mesh_y=DEFAULT_MESH_Y):
        """Plot training examples and separator hyperplane"""

        # Plot positive examples
        positive_examples = self.X_attributes[self.Y == 1]
        plt.plot(positive_examples[:, 0], positive_examples[:, 1], 'go')

        # Plot negative examples
        negative_examples = self.X_attributes[self.Y == -1]
        plt.plot(negative_examples[:, 0], negative_examples[:, 1], 'ro')

        # Plot separator hyperplane
        min_x, max_x = self.X_attributes[
            :, 0].min(), self.X_attributes[:, 0].max()
        min_y, max_y = self.X_attributes[
            :, 1].min(), self.X_attributes[:, 1].max()

        delta_x = max_x - min_x
        delta_y = max_y - min_y

        margin_x = DEFAULT_RELATIVE_MARGIN * delta_x
        margin_y = DEFAULT_RELATIVE_MARGIN * delta_y

        lin_x = np.linspace(min_x - margin_x, max_x + margin_x, mesh_x)
        lin_y = np.linspace(min_y - margin_y, max_y + margin_y, mesh_y)

        xx, yy = np.meshgrid(lin_x, lin_y)

        data_matrix = np.empty((mesh_x, mesh_y, 2))
        data_matrix[:, :, 0] = xx
        data_matrix[:, :, 1] = yy

        zz = np.empty((mesh_x, mesh_y))
        for i in range(zz.shape[0]):
            for j in range(zz.shape[1]):
                zz[i, j] = self.h_plot(data_matrix[i, j])

        plt.contour(xx, yy, zz, levels=[-1, 0, 1], colors=('r', 'b', 'g'),
                    linestyles = ('dashed', 'solid', 'dashed'))

        plt.grid(True)
        plt.show()
