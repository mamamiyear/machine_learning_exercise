# -*- coding: utf-8 -*-
__author__ = "mamamiyear"
import numpy as np
from random import random


def set_noise(v, noise):
    return v + (random() - 0.5) * 2 * noise


def create_feature(start, end, number, noise_max=0):
    standard_array = np.linspace(start, end, number, endpoint=True)
    noise_array = []
    for v in standard_array:
        noise_array.append(set_noise(v, noise_max))
    noise_array = np.array(noise_array)
    return noise_array


def create_target(theta0, theta1, feature_vector, noise_max=0):
    standard_array = theta0 + theta1 * feature_vector
    noise_array = []
    for v in standard_array:
        noise_array.append(set_noise(v, noise_max))
    noise_array = np.array(noise_array)
    return noise_array


def average_uniform(array):
    mu = np.average(array)
    max_value = np.max(array)
    min_value = np.min(array)
    uniform_array = (array - mu) / (max_value - min_value)
    return uniform_array


def min_uniform(array):
    max_value = np.max(array)
    min_value = np.min(array)
    uniform_array = (array - min_value) / (max_value - min_value)
    return uniform_array
