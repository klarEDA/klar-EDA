from os import makedirs
from os.path import join, exists
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .constants import VIZ_ROOT
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
from sklearn.manifold import TSNE


############################################################################
# To do: 1) resizing for the funnctions that require uniform size
#        2) handle rgb/gray images
#        3) axis labels, plot title
#        4) num components in eigen images
#        5) optimize mean/eigen computation
#        6) optimize std vs mean, different types of plots
#        7) object detection - plot x,y
#        8) batched feature extraction
############################################################################

class ImageDataVisualize:


    def __init__(self, data, labels, boxes=None):
        """Constructor for Image Data Visualization.

        :param data: images
        :type data: list of numpy image arrays
        :param labels: labels corresponding to each image
        :param boxes: list containing shape of each image
        """
        self.images = data
        self.labels = labels
        self.grey_present = False
        for image in self.images:
            if image.ndim < 3 or image.shape[-1] == 1:
                self.grey_present =True
                break
        # self.images = [np.expand_dims(image, axis=2) if image.ndim < 3 else image for image in self.images]
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



    def save_or_show(self, plot, plot_type, file_name,x_label=None, y_label=None, save=True, show=False):
        """To save the file(plot_type) in its designated directory or to make
        the path for the directory if such directory doesn't exist and then
        displaying the file type.

        :param plot: The figure to be plotted for graphical visualization.
        :type plot: png file.
        :param file_name: The filename to be stored.
        :type file_name: csv file
        :param x-label : The label to be put on the x-axis of the graph
        :type x-label: string
        :param y-label: The label to be put on the y-axis of the graph
        :type y-label: string
        :param save: To save the results in the background
        :type save: boolean
        :param show: To display the images in the foreground
        :type show: boolean
        """

        if save:
            save_dir = join(VIZ_ROOT, plot_type)
            if not exists(save_dir):
                makedirs(save_dir)
            if x_label != None:
                plt.xlabel(x_label)
            if y_label != None:
                plt.ylabel(y_label)
            plt.title("{}: {}".format(plot_type, file_name))
            save_path = join(save_dir, file_name)
            plot.savefig(save_path)
        if show:
            plt.title("{}: {}".format(plot_type, file_name))
            plt.show()
        plt.clf()




    def validate_images(self):
        """Function used to validate images, whether or not it has the required
        no of dimensions and whether it's a numpy array or not."""
        for image, label in zip(self.images, self.labels):
            if type(image) != np.ndarray:
                print('Image not a numpy array, skipping...')
                continue
            elif image.ndim < 2:
                print('Image has less than 2 dimensions, skipping...')
                self.images.remove(image)
                self.labels.remove(label)
                continue

                continue

    def aspect_ratio_histogram(self, save=True, show=False):
        """Function used to plot the aspect ratio histogram for the dataset.

        :param save: To save the results in the background
        :type save: boolean
        :param show: To display the images in the foreground
        :type show: boolean
        """
        aspect_ratios = self.dataset['Width'] / self.dataset['Height']
        plot = sns.histplot(aspect_ratios, bins='auto')



    def area_vs_category(self, save=True, show=False):
        """Function used to plot area per category of the images.

        :param save: To save the results in the background
        :type save: boolean
        :param show: To display the images in the foreground
        :type show: boolean
        """
        mean_areas = self.dataset.groupby('Label')['area'].mean()
        plot = sns.barplot(x=mean_areas.index, y=mean_areas.tolist())
        self.save_or_show(plot.figure, 'area_vs_category', 'area_vs_category', x_label='category',y_label= 'area', save=save, show=show)

    def mean_images(self, save=True, show=False):
        """Function used for evaluating the mean of the areas per category.

        :param save: To save the results in the background
        :type save: boolean
        :param show: To display the images in the foreground
        :type show: boolean
        """
        groups = self.dataset.groupby('Label')
        for group in groups:array(list(images)).mean(axis=0)
            plot = plt.imshow(mean_image/255)
            self.save_or_show(plot.figure, 'mean_images', str(group[0]), save=save, show=show)

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

                self.save_or_show(plot.figure, 'eigen_images/{}'.format(group[0]), str(i), save=save, show=show)

    def num_images_by_category(self, save=True, show=False):
        """Function used to display the no of images per category.

        :param save: To save the results in the background
        :type save: boolean
        :param show: To display the images in the foreground
        :type show: boolean
        """
        counts = self.dataset['Label'].value_counts()
        plot = sns.barplot(x=counts.index, y=counts.tolist())
        self.save_or_show(plt, 'num_images_by_category', 'pie_chart', save=save, show=show)

        self.save_or_show(plt, 'num_images_by_category', 'pie_chart', save=save, show=show)

    def std_vs_mean(self, save=True, show=False):
        """Function used to plot the graph of the standard deviation versus the
        mean plot.

        :param save: To save the results in the background
        :type save: boolean
        :param show: To display the images in the foreground
        :type show: boolean
        """
        groups = self.dataset.groupby('Label')
        y = [][]
        for group in groups:
            images = group[1]['Image']
            images = np.array(list(images))
            mean = images.mean()
            std = images.std()
            x.append(mean)
            y.append(std)
            hue.append(group[0])
        plot = sns.scatterplot(x=x, y=y, hue=hue, palette='viridis', legend='full')
        self.save_or_show(plot.figure, 'std_vs_mean', 'std_vs_mean_categories',x_label='mean', y_label='Std Deviation', save=save, show=show)

        means = self.dataset['Image'].apply(np.mean).to_list()
        stds = self.dataset['Image'].apply(np.std).to_list()
        labels = self.dataset['Label'].to_list()
        plot = sns.scatterplot(x=means, y=stds, hue=labels, palette='viridis', legend='full')
        self.save_or_show(plot.figure, 'std_vs_mean', 'std_vs_mean_all',x_label='mean', y_label='Std Deviation', save=save, show=show)


    def t_sne(self, batch_size=32, save=True, show=False):
        """ t-distributed Stochastic Neighbor Embedding - used to visualize high dimensional data
        
            :param batch_size: The size of the batch
            :type batch_size: integer
            :param save: To save the results in the background
            :type save: boolean
            :param show: To display the images in the foreground
            :type show: boolean
            """

        model = ResNet50(weights='imagenet', pooling=max, include_top = False)
        features_list = []
        print('Extracting features ...')
        for image in tqdm(self.images):
            if self.grey_present and (image.ndim < 3 or image.shape[-1] == 1):
                image = np.stack((image.squeeze(),)*3, axis=-1)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            features = model.predict(image)
            features_reduce = features.squeeze()
            features_list.append(features_reduce)

        print('Performing t-SNE ...')
        tsne = TSNE(n_components=2).fit_transform(features_list)
        x = tsne[:, 0]
        y = tsne[:, 1]
        x = (x - np.min(x)) / np.ptp(x)
        y = (y - np.min(y)) / np.ptp(y)

        plot = sns.scatterplot(x=x, y=y, hue=self.labels, palette='viridis', legend='full')
        self.save_or_show(plot.figure, 'tsne', 'tsne', x_label='Feature X', y_label='Feature Y', save=save, show=show)
