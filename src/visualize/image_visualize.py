from os import makedirs
from os.path import join, exists
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from constants import VIZ_ROOT
import cv2

############################################################################
# To do: 1) resize images before mean / eigen images
#        2) num components in eigen images
#        3) optimize mean/eigen computation
#        4) optimize std vs mean
#        5) object detection
############################################################################

class ImageDataVisualize:

    def __init__(self, data, labels, boxes=None):
        self.images = data
        self.labels = labels
        if len(self.images) != len(self.labels):
            raise ValueError('Number of images != Number of labels')
        self.validate_images()
        self.num_images = len(self.images)
        self.dataset = pd.DataFrame({
            'Image': self.images,
            'Height': [image.shape[0] for image in self.images] if not boxes
                        else [box[2] for box in boxes],
            'Width': [image.shape[1] for image in self.images] if not boxes
                        else [box[3] for box in boxes],
            'Label': self.labels,
        })
        self.dataset['area'] = self.dataset['Height'] * self.dataset['Width']
        print('Number of images after validation and filtering:', self.num_images)

    def save_or_show(self, plot, plot_type, file_name, save=True, show=False):
        if save:
            save_dir = join(VIZ_ROOT, plot_type)
            if not exists(save_dir):
                makedirs(save_dir)
            save_path = join(save_dir, file_name)
            plot.savefig(save_path)
        if show:
            plt.title("{}: {}".format(plot_type, file_name))
            plt.show()
        plt.clf()

    def validate_images(self):
        for image, label in zip(self.images, self.labels):
            if type(image) != np.ndarray:
                print('Image not a numpy array, skipping...')
                self.images.remove(image)
                self.labels.remove(label)
                continue
            elif image.ndim < 2:
                print('Image has less than 2 dimensions, skipping...')
                self.images.remove(image)
                self.labels.remove(label)
                continue

    def aspect_ratio_histogram(self, save=True, show=False):
        aspect_ratios = self.dataset['Width'] / self.dataset['Height']
        plot = sns.histplot(aspect_ratios, bins='auto')
        self.save_or_show(plot.figure, 'aspect_ratios', 'aspect_ratios', save=save, show=show)

    def area_vs_category(self, save=True, show=False):
        mean_areas = self.dataset.groupby('Label')['area'].mean()
        plot = sns.barplot(x=mean_areas.index, y=mean_areas.tolist())
        self.save_or_show(plot.figure, 'area_vs_category', 'area_vs_category', save=save, show=show)

    def mean_images(self, save=True, show=False):
        groups = self.dataset.groupby('Label')
        for group in groups:
            images = group[1]['Image']
            mean_image = np.array(list(images)).mean(axis=0)
            plot = plt.imshow(mean_image/255)
            self.save_or_show(plot.figure, 'mean_images', str(group[0]), save=save, show=show)

    def eigen_images(self, save=True, show=False):
        groups = self.dataset.groupby('Label')
        for group in groups:
            images = group[1]['Image']
            images = np.array(list(images))
            images = images.reshape(images.shape[0], -1)
            mean, eigenVectors = cv2.PCACompute(images, mean=None, maxComponents=10)
            eigenVectors = eigenVectors.reshape(10, 28, 28)
            mean = mean.reshape(28, 28)
            for i in range(10):
                # img = np.round(((((mean/255)*2)-1 + eigenVectors[i]) + 2) / 4)
                img = np.round((eigenVectors[i] + 1)/2)
                plot = plt.imshow(img)
                self.save_or_show(plot.figure, 'eigen_images/{}'.format(group[0]), str(i), save=save, show=show)

    def num_images_by_category(self, save=True, show=False):
        counts = self.dataset['Label'].value_counts()
        plot = sns.barplot(x=counts.index, y=counts.tolist())
        self.save_or_show(plot.figure, 'num_images_by_category', 'bar_chart', save=save, show=show)
        plot = plt.pie(counts.tolist(), labels=counts.index)
        self.save_or_show(plt, 'num_images_by_category', 'pie_chart', save=save, show=show)

    def std_vs_mean(self, save=True, show=False):
        groups = self.dataset.groupby('Label')
        y = []
        x = []
        hue = []
        for group in groups:
            images = group[1]['Image']
            images = np.array(list(images))
            mean = images.mean()
            std = images.std()
            x.append(mean)
            y.append(std)
            hue.append(group[0])
        plot = sns.scatterplot(x=x, y=y, hue=hue)
        self.save_or_show(plot.figure, 'std_vs_mean', 'std_vs_mean', save=save, show=show)
