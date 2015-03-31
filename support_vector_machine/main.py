#!/usr/bin/env python

import numpy as np
import svm


def main():
    datas = np.genfromtxt("datas.csv", delimiter=',')
    problem = svm.SupportVectorMachine(datas, 1000)

    problem.plot()


if __name__ == '__main__':
    main()
