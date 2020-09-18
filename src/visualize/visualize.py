from csv_visualize import CSVVisualize
from image_visualize import ImageDataVisualize
import tensorflow_datasets as tfds
import cv2
from random import randint
import numpy as np

def visualize_csv(csv_path):
    csvViz = CSVVisualize(csv_path)
    csvViz.plot_correlation_map()
    csvViz.plot_scatter_plots()
    csvViz.plot_grid_plots_for_categorical_features()
    csvViz.plot_horizontal_box_plot()
    csvViz.plot_regression_marginals()
    csvViz.plot_scatter_plot_matrix()
    csvViz.plot_paired_pointplots()
    csvViz.plot_scatter_plot_with_categorical()
    csvViz.plot_pie_chart()
    csvViz.plot_histogram()
    csvViz.plot_line_chart()
    csvViz.plot_diagonal_correlation_matrix()
    csvViz.plot_stem_plots()
    csvViz.plot_jitter_stripplot()
    print('Visualization Completed Successfully')
    
def test_csv_visualization():
    file_path = "/home/ask149/FOSSUnited/foss-hack-20/lib/titanic_data.csv" #add path to your test data
    visualize_csv(file_path)

def test_image_visualization_non_uniform_images():
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
    image_viz = ImageDataVisualize(images, labels)
    image_viz.aspect_ratio_histogram()
    image_viz.area_vs_category()
    image_viz.num_images_by_category()

def test_image_visualization_uniform_images():
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds = ds.take(1000)
    images = []
    labels = []
    for image, label in tfds.as_numpy(ds):
        images.append(image)
        labels.append(label)
    image_viz = ImageDataVisualize(images, labels)
    image_viz.mean_images()
    image_viz.eigen_images()
    image_viz.std_vs_mean()
    
test_csv_visualization()
# test_image_visualization_non_uniform_images()
# test_image_visualization_uniform_images()
