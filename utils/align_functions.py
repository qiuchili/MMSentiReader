# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:34:31 2019

@author: qiuchi
"""
import numpy as np

def dummy(intervals,features):
    return features

def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features