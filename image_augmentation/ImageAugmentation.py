#!/usr/bin/env python
# coding: utf-8
import random
import numpy as np
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
import warnings
warnings.filterwarnings("ignore")

# Generate random float values in desired range
def randomRange(a, b):
    return np.random.rand() * (b - a) + a

# Generate random crop in the center from the borders
def randomCrop(image):
    '''
    input: normalized image
    return: image croped 
    '''
    margin = 1/10
    A = [int(randomRange(0, image.shape[0] * margin)),
             int(randomRange(0, image.shape[1] * margin))]
    B = [int(randomRange(image.shape[0] * (1-margin), image.shape[0])), 
           int(randomRange(image.shape[1] * (1-margin), image.shape[1]))]
    return image[A[0]:B[0], A[1]:B[1]]

# Generate random intesity rescale
def randomIntensity(image):
    '''
    input: normalized image
    return: image with intensity reescale
    '''
    return rescale_intensity(image,
                             in_range=tuple(np.percentile(image, (randomRange(0,10), randomRange(90,100)))),
                             out_range=tuple(np.percentile(image, (randomRange(0,10), randomRange(90,100)))))

# Generate a random gamma
def randomGamma(image):
    '''
    input: normalized image
    return: image with random gama
    '''
    return adjust_gamma(image, gamma=randomRange(0.5, 1.5))

# Generate random gaussian bluring
def randomGaussian(image):
    '''
    input: normalized image
    return: image with gaussian blur
    '''
    return gaussian(image, sigma=randomRange(0, 5))
    
# Generate random noise
def randomNoise(image):
    '''
    input: normalized image
    return: image with random gaussian noise.
    '''
    return random_noise(image, var=randomRange(0.001, 0.01))

# Generate random filter     
def randomFilter(image):
    '''
    input: normalized image
    return: image with one of this filters
    [equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity]
    '''
    Filters = [equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity, flipUpsideDown]
    filt = random.choice(Filters)
    return filt(image)

# flip up-down using np.flipud
def flipUpsideDown(image):
    '''
    input: normalized image
    return: flippled image upside-down
    '''
    flipped= np.flipud(image)
    return flipped

# Generate random affine transformation wrapper
def randomAffine(image):
    '''
    input: normalized image
    return: image with random affine
    '''
    tform = AffineTransform(scale=(randomRange(0.75, 1.3), randomRange(0.75, 1.3)),
                            rotation=randomRange(-0.25, 0.25),
                            shear=randomRange(-0.2, 0.2),
                            translation=(randomRange(-image.shape[0]//10, image.shape[0]//10), 
                                         randomRange(-image.shape[1]//10, image.shape[1]//10)))
    return warp(image, tform.inverse, mode='reflect')

# Generate random perspective wrapper
def randomPerspective(image):
    '''
    input: normalized image
    return: image with random perspective
    '''
    region = 1/4
    Array1 = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]])
    Array2 = np.array([[int(randomRange(0, image.shape[1] * region)), int(randomRange(0, image.shape[0] * region))], 
                  [int(randomRange(0, image.shape[1] * region)), int(randomRange(image.shape[0] * (1-region), image.shape[0]))], 
                  [int(randomRange(image.shape[1] * (1-region), image.shape[1])), int(randomRange(image.shape[0] * (1-region), image.shape[0]))], 
                  [int(randomRange(image.shape[1] * (1-region), image.shape[1])), int(randomRange(0, image.shape[0] * region))], 
                 ])
    perspective_transformation = ProjectiveTransform()
    perspective_transformation.estimate(Array1, Array2)
    return warp(image, perspective_transformation, output_shape=image.shape[:2])

# Data augmentation from different approaches
def Augmentation(image, Changes=[randomAffine, randomFilter, randomNoise]):
    '''
    [randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop]
    input: normalized image
    return: perform image augmentation from various changes.
    '''
    for change in Changes:
        image = change(image)
    return image