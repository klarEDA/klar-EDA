from csv_preprocess import CSVPreProcess
from image_preprocess import ImagePreprocess
import cv2

def preprocess_csv(csv_path):
    csvPreP = CSVPreProcess(csv_path)
    #Append preprocessing functions here
    print('CSV Preprocessing completed successfully!')

def preprocess_images(data_path, dataset_type):
    imagePrep = ImagePreprocess(data_path)
    imagePrep.resize_images(height = 512, width = 512)
    if dataset_type == 'ocr':           
        imagePrep.denoise()
        imagePrep.colorize(text = True)
        imagePrep.thresholding(technique = 'gaussian' ,threshold = cv2.THRESH_BINARY)
        # imagePrep.erode()
        # imagePrep.dilation()
    elif dataset_type == 'face':
        imagePrep.detect_face_and_crop(crop=True)
        imagePrep.colorize(text = False)
        imagePrep.adaptive_histogram_equalization()
        imagePrep.denoise(is_gray=True)
        imagePrep.normalize()
        imagePrep.erode()
        imagePrep.dilation()
        imagePrep.contrast_control()
    else:
        imagePrep.colorize(text = False)
        imagePrep.adaptive_histogram_equalization()
        imagePrep.normalize()
        imagePrep.denoise(is_gray=True)
        imagePrep.erode()
        imagePrep.dilation()
        imagePrep.contrast_control()
    #Append preprocessing functions here
    print('Image Preprocessing completed successfully!')
    
def test_csv_preprocessing():
    file_path = "" #add path to your test data
    preprocess_csv(file_path)

def test_image_preprocessing():
    dataset_path = "" #add path to your test data
    dataset_type = ''
    preprocess_images(dataset_path, dataset_type)

test_csv_preprocessing()
test_image_preprocessing()