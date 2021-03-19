import os
from os import makedirs
from os.path import join, exists
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ..image_preprocess import ImagePreprocess

class IntelligentImagePreprocess:
    """
    This class contains the functions:
    
    """    
    def __init__(self,input,labels = None):
        self.suffixes = ('.jpeg', '.jpg', '.png')
        # self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.labels = labels
        if type(input)==str:
            self.path = input
            self.image_list = sorted([ file for file in os.listdir(input) if (file.endswith(self.suffixes))])
            self.cv2_image_list = [ self.read_images(os.path.join(self.path,image_name)) for image_name in  self.image_list ]
        else:
            self.path = None
            self.image_list = None
            self.cv2_image_list = input
    
    #  the functions
    def detect_face_and_crop(self, crop = False, save=True, show=False):
        face_image_list = []
        image_index = -1
        face_cascade = self.get_cascade('face')
        for image in self.cv2_image_list:
            try:
                image_index += 1
                img = image.copy()
                faces = face_cascade.detectMultiScale(img, 1.3, 5)
                if faces is None:
                    print('Unable to find face ')
                    continue
                for (x,y,w,h) in faces:
                    padding = 10
                    ih, iw = img.shape[:2]
                    lx = max( 0, x - padding )
                    ly = max( 0, x - padding )
                    ux = min( iw, x + w + padding )
                    uy = min( ih, y + h + padding )
                    img = cv2.rectangle(img,(lx,ly),(ux,uy),(255,0,0),2)
                    roi_color = img[y:y+h, x:x+w]
                    if crop == True:
                        self.save_or_show_image(roi_color, image_index, 'haarcascade_faces',save=save,show=show)
                self.save_or_show_image(img, image_index, 'haarcascade',save=save,show=show)
                face_image_list.append(img)
            except Exception as e:
                print('Error while detecing')
        self.cv2_image_list = face_image_list