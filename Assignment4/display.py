# display.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import time
import numpy as np
import skimage.io as io
import skimage.transform as transform


def displayDigit(data, nrows=1, outfile='sample.png'):
    assert data.shape[1] == 784
    assert np.max(data)<=1 and np.min(data)>=0
    ncols = int(np.ceil(len(data)/nrows))
    # resize to 56*56
    data = data.reshape([-1,28,28])
    output = np.zeros((90*nrows, 90*ncols))
    for (n, d) in enumerate(data):
        d = transform.resize(d, (84,84))
        i = n // ncols
        j = n % ncols
        output[i*90+3:i*90+87,j*90+3:j*90+87] = d
    io.imsave(outfile, output)