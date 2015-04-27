#!/usr/bin/env python

import numpy as np
import svm


def main():
    datas = np.genfromtxt("datas_separable.csv", delimiter=',')
    problem = svm.SupportVectorMachine(datas, 1000)

    problem.train()
    problem.plot()

    datas = np.genfromtxt("datas_not_separable.csv", delimiter=',')
    problem = svm.SupportVectorMachine(datas, 1000, 1)

    problem.train()
    problem.plot()

if __name__ == '__main__':
    main()
