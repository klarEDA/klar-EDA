import os
from os import makedirs
from os.path import join, exists
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ..image_preprocess import ImagePreprocess

"""
This document contains the functions:
Thresholding, Denoise, Erode, Dilation, normalize
"""

class MorphologicalPreprocess:
    def __init__(self,input,labels = None):
        self.suffixes = ('.jpeg', '.jpg', '.png')
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.labels = labels
        if type(input)==str:
            self.path = input
            self.image_list = sorted([ file for file in os.listdir(input) if (file.endswith(self.suffixes))])
            self.cv2_image_list = [ self.read_images(os.path.join(self.path,image_name)) for image_name in  self.image_list ]
        else:
            self.path = None
            self.image_list = None
            self.cv2_image_list = input
    def thresholding(self, technique = 'mean', threshold = cv2.THRESH_BINARY, save=True, show=False):
        binarized_image_list = []
        image_index = 0
        #study the parameters
        for image in self.cv2_image_list:
            try:
                if technique == 'simple':
                    res , img = cv2.threshold(image, 120, 255, threshold)
                    binarized_image_list.append(img)
                    ImagePreprocess.save_or_show_image(img,image_index,'threshold',save=save,show=show)
                    image_index += 1
                elif technique == 'mean':
                    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold, 199, 5)
                    binarized_image_list.append(img)
                    ImagePreprocess.save_or_show_image(img,image_index,'threshold',save=save,show=show)
                    image_index += 1
                else:
                    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold, 199, 5)
                    binarized_image_list.append(img)
                    ImagePreprocess.save_or_show_image(img,image_index,'threshold',save=save,show=show)
                    image_index += 1
            except Exception as e:
                print('Error during binarization of image ', image_index, e)
        self.cv2_image_list = binarized_image_list
    
    def denoise(self, is_gray = True, save=True, show=False):
        denoised_image_list = []
        image_index = 0
        for image in self.cv2_image_list:
            try:
                if not is_gray:
                    img = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
                else:
                    img = cv2.fastNlMeansDenoising(image,None,3,7,21)
                denoised_image_list.append(img)
                ImagePreprocess.save_or_show_image(img,image_index,'denoise',save=save,show=show)
                image_index += 1
            except Exception as e:
                print('Error during denoising image ', image_index, e)
        self.cv2_image_list = denoised_image_list
        
    def erode(self, dim = None, save=True, show=False):
        eroded_image_list = []
        image_index = 0
        if dim == None:
            dim = (2,2)
        for image in self.cv2_image_list:
            try:
                kernel = np.ones(dim,np.uint8)
                img = cv2.erode(image,kernel,iterations = 1)
                ImagePreprocess.save_or_show_image(img,image_index,'erode',save=save,show=show)
                image_index += 1
                eroded_image_list.append(img)
            except Exception as e:
                print('Error during eroding image ', image_index, e)
        self.cv2_image_list = eroded_image_list

    def dilation(self, dim = None, save=True, show=False):
        dilated_image_list = []
        image_index = 0
        if dim == None:
            dim = (2,2)
        for image in self.cv2_image_list:
            try:
                kernel = np.ones(dim,np.uint8)
                img = cv2.dilate(image,kernel,iterations = 1)
                ImagePreprocess.save_or_show_image(img,image_index,'dilation',save=save,show=show)
                image_index += 1
                dilated_image_list.append(img)
            except Exception as e:
                print('Error while dilating image ', image_index, e)
        self.cv2_image_list = dilated_image_list
        
    def normalize(self, dim = None, save=True, show=False):
        normalized_image_list = []
        image_index = 0
        if dim == None:
            dim = (512,512)
        for image in self.cv2_image_list:
            try:
                kernel = np.zeros(dim)
                img = cv2.normalize(image,kernel,0,255,cv2.NORM_MINMAX)
                normalized_image_list.append(img)
                ImagePreprocess.save_or_show_image(img,image_index,'normalize',save=save,show=show)
                image_index += 1
            except Exception as e:
                print('Error while normalizing image ', image_index, e)