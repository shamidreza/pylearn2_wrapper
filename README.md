pylearn2_wrapper
==============
Author: Seyed Hamidreza Mohammadi (http://www.cslu.ogi.edu/~mohammah/)

Date: 2014-05-13 (http://www.w3.org/TR/NOTE-datetime)

This is simple wrapper/script for using pylearn2.
It interfaces data through numpy arrays instead of Theano objects, which makes it easier if you don't want to deal with Theano objects.

Contains the following classes:

    RegularAutoencoder  
  
    DenoisingAutoencoder
  
    ContractiveAutoencoer
  
    HigherOrderContractiveAutoencoer
  
    DeepGeneralAutoencoder
  
    MLP (Multilayer Perceptron)
  
  
Dependencies:

    1- NumPy: http://www.scipy.org/scipylib/download.html

    2- Theano (bleeding-edge version): http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions

    3- Pylearn2: http://deeplearning.net/software/pylearn2/

Cite:

S.H. Mohammadi, A. Kain, Voice Conversion Using Deep Neural Networks With Speaker-Independent Pre-Training, 2014 IEEE Spoken Language Technology Workshop (SLT), 2014.
