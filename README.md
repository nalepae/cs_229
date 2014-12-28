Machine Learning
================

This repository contains **Octave** scripts.
On **Ubuntu**, to install **Octave**, just write in a terminal :

**sudo apt-get install octave**

-------

For each directories, choose parameters in section *PARAMETERS* in the beginning of file *main.m*.
Execute this file with the following command : *octave main.m*.

-------

List of parameters :

* DATA_FILE : The path to data file containing training examples
* ALPHA : Learning rate
* DEGREE : Degree of regression (1 for linear, 2 for quadratic, 3 for cubic ...)
* LAST_ITERATION : The number of iteration for the regression
* LAMBDA : Regularization parameter

List of plot parameters :

* NUM_THETA_X : The theta's number to be printed on x axis
* NUM_THETA_Y : The theta's number to be printes on y axis