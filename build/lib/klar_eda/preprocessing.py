from .preprocess.csv_preprocess import CSVPreProcess
from .preprocess.image_preprocess import ImagePreprocess

def preprocess_csv(csv, target_column=None, index_column=None):
    preprocessor = CSVPreProcess(csv, target_col=target_column, index_column=index_column)
    preprocessor.encode_categorical_target()
    preprocessor.fill_numerical_na()
    preprocessor.fill_categorical_na()
    preprocessor.encode_categorical()
    preprocessor.remove_outliers()
    preprocessor.normalize_numerical()
    preprocessor.remove_non_contributing_features()

def preprocess_images(data_path, dataset_type):
    preprocessor = ImagePreprocess(data_path)
    preprocessor.resize_images(height = 512, width = 512)
    if dataset_type == 'ocr':           
        preprocessor.denoise()
        preprocessor.colorize(text = True)
        preprocessor.thresholding(technique = 'gaussian' ,threshold = cv2.THRESH_BINARY)
    elif dataset_type == 'face':
        preprocessor.detect_face_and_crop(crop=True)
        preprocessor.colorize(text = False)
        preprocessor.adaptive_histogram_equalization()
        preprocessor.denoise(is_gray=True)
        preprocessor.normalize()
        preprocessor.erode()
        preprocessor.dilation()
        preprocessor.contrast_control()
    else:
        preprocessor.colorize(text = False)
        preprocessor.adaptive_histogram_equalization()
        preprocessor.normalize()
        preprocessor.denoise(is_gray=True)
        preprocessor.erode()
        preprocessor.dilation()
        preprocessor.contrast_control()
    print('Image Preprocessing completed successfully!')