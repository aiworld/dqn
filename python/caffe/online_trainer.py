#!/usr/bin/env python
"""
Online trainer takes images one at a time and make predictions as you go.
"""

import numpy as np

import caffe


class OnlineTrainer(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    """
    def __init__(self, model_file):
        caffe.Net.__init__(self, model_file)
        # caffe.Net.__init__(self, 'examples/dqn/dqn_train.prototxt')

