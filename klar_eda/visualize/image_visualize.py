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
   """init:this function is for initializing the parameters to work on
      :self_param:the file from which we have to takee the data to work on
      :self_type:csv file
      :data_param:the images form our dataset
      :labels_param:to categorize the images.
      :boxes_param:a null parameter used for storing the dimensions of the images."""
  
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
  """function save_or_show: to save the file(plot_type) in its designated directory or to make the path for the directory if such directory doesn't exist and then displaying the file type.
     :plot_param-the figure to be plotted for graphical visualization. 
     :plot_type- the file in which all the visualizations are stored.
     :file_name- the name of the file to be stored.
     :x-label - the label to be put on the x-axis of the graph
     :y-label- the label to be put on the y-axis of the graph
     :save-parameter- the boolean parameter passed for saving the file.
     :show-parameter- display the fiel along with its title and also displayin gthe plot."""
    
  
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
        """validate_images:the function used to validate images,whether or not  it has the required no of dimensions  and whether  it's a numpy array or not.
            :self-the dataset on which the visualization and the analysis has to be performed."""
        
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
        """ aspect_ratio_histogram:the function used to define the aspect ratio of the histogram.aspect_ratio=Width/Height.
            :save-param:the boolean  for instructing  to save the file.
            :show-param:to display the ratios,the plot ,the labels and everything related to visualisation."""
        aspect_ratios = self.dataset['Width'] / self.dataset['Height']
        plot = sns.histplot(aspect_ratios, bins='auto')
        self.save_or_show(plot.figure, 'aspect_ratios', 'aspect_ratios', x_label='aspect_ratios', y_label='frequency', save=save, show=show)


    def area_vs_category(self, save=True, show=False):
        """area_vs_category:the plot to show the areas percategory(label).
            :save-param:the boolean  for instructing  to save the file.
            :show-param:to display the ratios,the plot ,the labels and everything related to visualisation."""
        mean_areas = self.dataset.groupby('Label')['area'].mean()
        plot = sns.barplot(x=mean_areas.index, y=mean_areas.tolist())
        self.save_or_show(plot.figure, 'area_vs_category', 'area_vs_category', x_label='category',y_label= 'area', save=save, show=show)

    def mean_images(self, save=True, show=False):
        """mean_images:The function for evaluating the mean of the areas per category.
            :save-param:the boolean  for instructing  to save the file.
            :show-param:to display the ratios,the plot ,the labels and everything related to visualisation."""
        groups = self.dataset.groupby('Label')
        for group in groups:
            images = group[1]['Image']
            mean_image = np.array(list(images)).mean(axis=0)
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

    def num_images_by_category(self, save=True, show=False):
        """ the function to display the no of images per category.
            :save-param:the boolean  for instructing  to save the file.
            :show-param:to display the ratios,the plot ,the labels and everything related to visualisation."""
        counts = self.dataset['Label'].value_counts()
        plot = sns.barplot(x=counts.index, y=counts.tolist())
        self.save_or_show(plot.figure, 'num_images_by_category', 'bar_chart',x_label='category', y_label='No. of images', save=save, show=show)
        plot = plt.pie(counts.tolist(), labels=counts.index)
        self.save_or_show(plt, 'num_images_by_category', 'pie_chart', save=save, show=show)

    def std_vs_mean(self, save=True, show=False):
        """std_vs_mean:the function used to plot the graph of the standard deviation versus the mean.
          : self_param:The dataset on which the analysis is used.
          :save-param:the boolean  for instructing  to save the file.
          :show-param:to display the ratios,the plot ,the labels and everything related to visualisation."""
        
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
        plot = sns.scatterplot(x=x, y=y, hue=hue, palette='viridis', legend='full')
        self.save_or_show(plot.figure, 'std_vs_mean', 'std_vs_mean_categories',x_label='mean', y_label='Std Deviation', save=save, show=show)

        means = self.dataset['Image'].apply(np.mean).to_list()
        stds = self.dataset['Image'].apply(np.std).to_list()
        labels = self.dataset['Label'].to_list()
        plot = sns.scatterplot(x=means, y=stds, hue=labels, palette='viridis', legend='full')
        self.save_or_show(plot.figure, 'std_vs_mean', 'std_vs_mean_all',x_label='mean', y_label='Std Deviation', save=save, show=show)


    def t_sne(self, batch_size=32, save=True, show=False):
        """t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space.
            :batch_size:the dataset is dividied into batches for smooth functioning of the model on the dataset.
            :save-param:the boolean  for instructing  to save the file.
            ::show-param:to display the ratios,the plot ,the labels and everything related to visualisation."""
        
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
