import os
from os import makedirs
from os.path import join, exists
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from constants import VIZ_ROOT

class ImagePreprocess:

    def __init__(self,path):
        self.path = path
        self.suffixes = ('.jpeg', '.jpg', '.png')
        self.image_list = sorted([ file for file in os.listdir(path) if (file.endswith(self.suffixes))])
        self.cv2_image_list = [ self.read_images(os.path.join(self.path,image_name)) for image_name in  self.image_list ]
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def read_images(self, image_path):
        img = cv2.imread(image_path)
        return img

    def save_or_show_image(self, image, image_name, processing_type, save=True, show=False):
        if save:
            save_dir = join(VIZ_ROOT, processing_type)
            if not exists(save_dir):
                makedirs(save_dir)
            image_name = 'img'+str(image_name)+'.jpg'
            save_path = join(save_dir, image_name)
            cv2.imwrite(save_path,image)
        if show:
            cv2.imshow(str(image_name),image)
            cv2.waitKey()
        cv2.destroyAllWindows()

    def get_interpolation_technique(self, img, dim, inter = None):
        if inter == None:
            ( h, w ) = img.shape[:2]
            if h < dim[1] and w < dim[0]:
                return cv2.INTER_LINEAR
            if h > dim[1] and w > dim[0]:
                return cv2.INTER_AREA
            return cv2.INTER_CUBIC
        return inter
    
    def resize_images(self, height = 512, width = 512, inter = None):
        resized_image_list = []
        dim = ( width, height )
        image_index = 0
        for image in self.cv2_image_list:
            #default interpolation used is cv.INTER_LINEAR
            img = cv2.resize(image, dim, interpolation = self.get_interpolation_technique( image, dim, inter))
            resized_image_list.append(img)
            self.save_or_show_image(img,image_index,'resize')
            image_index += 1
        self.cv2_image_list = resized_image_list

    def colorize(self, color = None, text = False):
        colorized_image_list = []
        image_index = 0
        # add some smart logic based on the type of dataset ( face, text, scenary, etc.)
        if color == None:
            color = cv2.COLOR_BGR2GRAY
        for image in self.cv2_image_list:
            img = cv2.cvtColor(image, color)
            # if text == True:
            #     img = cv2.bitwise_not(img)
            colorized_image_list.append(img)
            self.save_or_show_image(img,image_index,'colorize')
            image_index += 1
        self.cv2_image_list = colorized_image_list

    def contrast_control(self, alpha = 1.25, beta = 0):
        contrast_image_list = []
        image_index = 0
        for image in self.cv2_image_list:
            np_image = image.copy().astype(float)
            img = cv2.convertScaleAbs(np_image, alpha = alpha, beta = beta)
            contrast_image_list.append(img)
            self.save_or_show_image(img,image_index,'contrast')
            image_index += 1
        self.cv2_image_list = contrast_image_list

    def thresholding(self, technique = 'mean', threshold = cv2.THRESH_BINARY):
        binarized_image_list = []
        image_index = 0
        #study the parameters
        for image in self.cv2_image_list:
            if technique == 'simple':
                res , img = cv2.threshold(image, 120, 255, threshold)
                binarized_image_list.append(img)
                self.save_or_show_image(img,image_index,'threshold')
                image_index += 1
            elif technique == 'mean':
                img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold, 199, 5)
                binarized_image_list.append(img)
                self.save_or_show_image(img,image_index,'threshold')
                image_index += 1
            else:
                img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold, 199, 5)
                binarized_image_list.append(img)
                self.save_or_show_image(img,image_index,'threshold')
                image_index += 1
        self.cv2_image_list = binarized_image_list
    
    def denoise(self, is_gray = True):
        denoised_image_list = []
        image_index = 0
        for image in self.cv2_image_list:
            if not is_gray:
                img = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
            else:
                img = cv2.fastNlMeansDenoising(image,None,3,7,21)
            denoised_image_list.append(img)
            self.save_or_show_image(img,image_index,'denoise')
            image_index += 1
        self.cv2_image_list = denoised_image_list
        
    def erode(self, dim = None):
        eroded_image_list = []
        image_index = 0
        if dim == None:
            dim = (2,2)
        for image in self.cv2_image_list:
            kernel = np.ones(dim,np.uint8)
            img = cv2.erode(image,kernel,iterations = 1)
            self.save_or_show_image(img,image_index,'erode')
            image_index += 1
            eroded_image_list.append(img)
        self.cv2_image_list = eroded_image_list

    def dilation(self, dim = None):
        dilated_image_list = []
        image_index = 0
        if dim == None:
            dim = (2,2)
        for image in self.cv2_image_list:
            kernel = np.ones(dim,np.uint8)
            img = cv2.dilate(image,kernel,iterations = 1)
            self.save_or_show_image(img,image_index,'dilation')
            image_index += 1
            dilated_image_list.append(img)
        self.cv2_image_list = dilated_image_list
        
    def normalize(self, dim = None):
        normalized_image_list = []
        image_index = 0
        if dim == None:
            dim = (512,512)
        for image in self.cv2_image_list:
            kernel = np.zeros(dim)
            img = cv2.normalize(image,kernel,0,255,cv2.NORM_MINMAX)
            normalized_image_list.append(img)
            self.save_or_show_image(img,image_index,'normalize')
            image_index += 1

    def print_variables(self):
        for img in self.cv2_image_list:
            cv2.imshow('img',img)
            cv2.waitKey()
    
    def get_cascade(self, cascade_type='face'):
        #if cascade_type == 'face':
        return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect_face_and_crop(self, crop = False):
        face_image_list = []
        image_index = -1
        face_cascade = self.get_cascade('face')
        for image in self.cv2_image_list:
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
                    self.save_or_show_image(roi_color, image_index, 'haarcascade_faces')
            self.save_or_show_image(img, image_index, 'haarcascade')
            face_image_list.append(img)
        self.cv2_image_list = face_image_list

    def adaptive_histogram_equalization(self):
        refined_image_list = []
        image_index = 0
        for image in self.cv2_image_list:
            try:
                img = self.clahe.apply(image)
                refined_image_list.append(img)
                self.save_or_show_image(img, image_index, 'clahe')
                image_index += 1
            except Exception as e:
                print('Error during adaptive histogram equalization' , e)
        self.cv2_image_list = refined_image_list
