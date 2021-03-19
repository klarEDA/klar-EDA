from .preprocess.csv_preprocess import CSVPreProcess
from .preprocess.image_preprocess import ImagePreprocess
from .preprocess.image_preprocess.morphological import MorphologicalPreprocess

def preprocess_csv(csv, target_column=None, index_column=None):
    """Preprocesses the csv file OR the dataframe, 
    generates a csv file in the current directory with preprocessed data - Preprocess_file.csv

    :param csv: Either pandas Dataframe ( with column names as row 0 ) OR path to csv file
    :type csv: pandas.Dataframe / string
    :param target_column: Name of the target column, defaults to last column in the dataframe.
    :type target_column: string, optional
    :param index_column: List of column names which contain indexes/ do not contribute at all, defaults to None
    :type index_column: list of string, optional
    """
    preprocessor = CSVPreProcess(csv, target_col=target_column, index_column=index_column)
    preprocessor.encode_categorical_target()
    preprocessor.fill_numerical_na()
    preprocessor.fill_categorical_na()
    preprocessor.encode_categorical()
    preprocessor.remove_outliers()
    preprocessor.normalize_numerical()
    preprocessor.remove_non_contributing_features()
    print('CSV Preprocessing completed successfully!')

def preprocess_images(data_path, dataset_type='other',save=True,show=False):
    """Processes the image data, and generates folders with preprocessed images.

    :param data_path: Path to folder containing image data ( Caution : Make sure the folder contains only images )
    :type data_path: string
    :param dataset_type:  Either 'ocr' , 'face' or 'other' - Preprocessing is different for each category, defaults to 'other'
    :type dataset_type: string, optional
    :param save: Save the results to directory, defaults to True
    :type save: bool, optional
    :param show: Preview the results, defaults to False
    :type show: bool, optional
    """
    preprocessor = ImagePreprocess(data_path)
    morphPreprocessor = MorphologicalPreprocess(data_path)
    
    preprocessor.resize_images(height = 512, width = 512)
    if dataset_type == 'ocr':
        morphPreprocessor.denoise(save=save,show=show)
        preprocessor.colorize(text = True,save=save,show=show)
        morphPreprocessor.thresholding(technique = 'gaussian' ,threshold = cv2.THRESH_BINARY,save=save,show=show)
    elif dataset_type == 'face':
        preprocessor.detect_face_and_crop(crop=True,save=save,show=show)
        preprocessor.colorize(text = False,save=save,show=show)
        preprocessor.adaptive_histogram_equalization(save=save,show=show)
        morphPreprocessor.denoise(is_gray=True,save=save,show=show)
        morphPreprocessor.normalize(save=save,show=show)
        morphPreprocessor.erode(save=save,show=show)
        morphPreprocessor.dilation(save=save,show=show)
        preprocessor.contrast_control(save=save,show=show)
    else:
        preprocessor.colorize(text = False,save=save,show=show)
        preprocessor.adaptive_histogram_equalization(save=save,show=show)
        morphPreprocessor.normalize(save=save,show=show)
        morphPreprocessor.denoise(is_gray=True,save=save,show=show)
        morphPreprocessor.erode(save=save,show=show)
        morphPreprocessor.dilation(save=save,show=show)
        preprocessor.contrast_control(save=save,show=show)
    print('Image Preprocessing completed successfully!')