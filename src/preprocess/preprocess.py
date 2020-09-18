from csv_preprocess import CSVPreProcess
from image_preprocess import ImagePreprocess
import cv2
from random import randint
import tensorflow_datasets as tfds

def preprocess_csv(csv_path):
    csvPreP = CSVPreProcess(csv_path)
    #Append preprocessing functions here
    print('CSV Preprocessing completed successfully!')

def get_sample_data():
    ds = tfds.load('cifar10', split='train', as_supervised=True)
    ds = ds.take(1000)
    images = []
    labels = []
    for image, label in tfds.as_numpy(ds):
        h = randint(24, 56)
        w = randint(24, 56)
        image = cv2.resize(image, (w, h))
        images.append(image)
        labels.append(label)
    image_prep = ImagePreprocess(images, labels)
    return image_prep


def preprocess_images(data_path, dataset_type):
    image_prep = ImagePreprocess(data_path)
    image_prep.resize_images(height = 512, width = 512)
    if dataset_type == 'ocr':           
        image_prep.denoise()
        image_prep.colorize(text = True)
        image_prep.thresholding(technique = 'gaussian' ,threshold = cv2.THRESH_BINARY)
        # image_prep.erode()
        # image_prep.dilation()
    elif dataset_type == 'face':
        image_prep.detect_face_and_crop(crop=True)
        image_prep.colorize(text = False)
        image_prep.adaptive_histogram_equalization()
        image_prep.denoise(is_gray=True)
        image_prep.normalize()
        image_prep.erode()
        image_prep.dilation()
        image_prep.contrast_control()
    else:
        image_prep.colorize(text = False)
        image_prep.adaptive_histogram_equalization()
        image_prep.normalize()
        image_prep.denoise(is_gray=True)
        image_prep.erode()
        image_prep.dilation()
        image_prep.contrast_control()
    #Append preprocessing functions here
    print('Image Preprocessing completed successfully!')
    
def test_csv_preprocessing():
    file_path = "" #add path to your test data
    preprocess_csv(file_path)

def test_image_preprocessing():
    dataset_path = "" #add path to your test data
    dataset_type = "ocr" # ocr/ face/ default
    preprocess_images(dataset_path, dataset_type)

test_csv_preprocessing()
test_image_preprocessing()