Machine Learning
================

This repository contains **Octave** scripts.
On **Ubuntu**, to install **Octave**, just write in a terminal :

**sudo apt-get install octave**

-------

Directory **linear_regression** :

File **linear_regression_1_feature_real_time.m** :

Choose in the header of file :
* Learning rate **ALPHA**
* Initial learning parameters **INITIAL_THETA**
* Number of iterations **NB_ITERATION**

Execute this file with **Octave**.

**Octave** will plot in real time, by using the gradient descent technique :
* The regression's cost
* Data set
* Regressed curve
* 3D curve of cost in term of learning parameters
* Learning parameters on iso-cost curves

File **linear_regression_1_feature_final_state.m** :

Choose in the header of file :
* Learning rate **ALPHA**
* Initial learning parameters **INITIAL_THETA**
* Number of iterations **NB_ITERATION**

Execute this file with **Octave**.

**Octave** will plot for the last iteration of gradient descent technique :
* The regression's cost
* Data set
* Regressed curve
* 3D curve of cost in term of learning parameters
* Learning parameters on iso-cost curves

File **linear_regression_1_feature_polynomial_real_time.m** :

Choose in the header of file :
* Learning rate **ALPHA**
* Polynomial degree **DEGREE**
* Number of iterations **NB_ITERATION**

Execute this file with **Octave**.

**Octave** will plot in real time, by using the gradient descent technique :
* The regression's cost
* Data set
* Regressed curve

File **linear_regression_1_feature_polynomial_final_state.m** :

Choose in the header of file :
* Learning rate **ALPHA**
* Polynomial degree **DEGREE**
* Number of iterations **NB_ITERATION**

Execute this file with **Octave**.

**Octave** will plot for the last iteration of gradient descent technique :
* The regression's cost history
* Data set
* Regressed curve

File **normal_equation_1_feature_polynomial.m** :

Choose in the header of file :
* Polynomial degree **DEGREE**

Execute this file with **Octave**.

**Octave** will plot by using the normal equation technique :
* Data set
* Regressed curve
