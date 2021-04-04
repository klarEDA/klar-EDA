from os import makedirs
from os.path import join, exists
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from .constants import VIZ_ROOT, NUNIQUE_THRESHOLD
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,join(parent_dir,'preprocess'))
from ..preprocess.csv_preprocess import CSVPreProcess

class CSVVisualize:
    """This class generates visualization plots, graphs and charts of data from csv file

    :param input: Either a path to csv file or a pandas DataFrame
    :type input: str/ pandas.DataFrame
    :param target_col: Name of the target column, defaults to None
    :type target_col: str, optional
    :param index_column: Either column index or string name of column(s) to be used as index 
        of the DataFrame, defaults to None
    :type index_column: int/ str/ list, optional
    :param exclude_columns: List of columns to be excluded from visualization, defaults to []
    :type exclude_columns: list, optional
    """
    def __init__(self, input, target_col = None, index_column = None, exclude_columns = []):
        """Parameterized Constructor to initialize data members
        """
        if type(input)==str:
            self.df = pd.read_csv(input, index_col = index_column)
        else:
            self.df = input
        self.df.drop(exclude_columns,inplace=True)
        self.col_names = list(self.df.columns)
        self.target_column = self.col_names[-1] if target_col == None else target_col
        self.df.dropna(subset=[self.target_column], inplace=True)
        self.num_cols = len(self.col_names)
        self.output_format = 'png'
        self.categorical_data_types = ['object','str']
        viz = CSVPreProcess(self.df, target_col = target_col, index_column = index_column)
        self.df = viz.fill_numerical_na(ret = True)
        self.df = viz.fill_categorical_na(ret = True)
        self.categorical_column_list = []
        self.populate_categorical_column_list()
        self.numerical_column_list = list(self.get_filtered_dataframe(include_type=np.number))
        temp_col_list = [num_col for num_col in self.numerical_column_list if self.df[num_col].nunique() < NUNIQUE_THRESHOLD]
        self.continuous_column_list = [x for x in self.numerical_column_list if x not in temp_col_list]
        self.non_continuous_col_list = self.categorical_column_list + temp_col_list

    def save_or_show(self, plot, plot_type, file_name, x_label=None, y_label=None, save=True, show=False):
        """This method saves the plot as figure and/or shows(displays) the plot depending on the arguments passed

        :param plot: Figure object used to save the plot as figure
        :type plot: figure object
        :param plot_type: Type of plot generated
        :type plot_type: str
        :param file_name: Name of file in which the figure is to be saved 
        :type file_name: str
        :param x_label: Label text for x-axis of the plot, defaults to None
        :type x_label: str, optional
        :param y_label: Label text for y-axis of the plot, defaults to None
        :type y_label: str, optional
        :param save: 'True' if plot is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if plot is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        if save:
            save_dir = join(VIZ_ROOT, plot_type)
            if not exists(save_dir):
                makedirs(save_dir)
            save_path = join(save_dir, file_name)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title("{}: {}".format(plot_type, file_name))
            plot.savefig(save_path)
        if show:
            plt.title("{}: {}".format(plot_type, file_name))
            plt.show()
        plt.clf()

    def get_filtered_dataframe(self, include_type = [], exclude_type = []):
        """This method filters the DataFrame by returning a subset of the DataFrame's columns depending on 
        the dtypes to be included/excluded. 

        :param include_type: dtypes to be included while filtering, defaults to []
        :type include_type: list, optional
        :param exclude_type: dtypes to be excluded while filtering, defaults to []
        :type exclude_type: list, optional
        :return: Original DataFrame if both include_type and exclude_type are empty. Otherwise, a subset 
            of the DataFrame's columns based on the dtypes to be included/excluded
        :rtype: pandas.DataFrame
        """
        if include_type or exclude_type:
            return self.df.select_dtypes(include = include_type, exclude = exclude_type)
        else:
            return self.df

    def populate_categorical_column_list(self):
        """This method populates the categorical column list by appending the columns in the filtered DataFrame 
        with distinct observations less than or equal to the NUNIQUE_THRESHOLD to it.
        """
        df = self.get_filtered_dataframe(exclude_type=np.number)
        if not self.categorical_column_list:
            for column in df:
                if df[column].nunique() <= NUNIQUE_THRESHOLD:
                    self.categorical_column_list.append(column)

    def get_categorical_numerical_columns_pairs(self):
        """This method generates a list of paired columns of categorical data taking on numerical values

        :return: List of paired columns
        :rtype: list
        """
        categorical_column_list = self.categorical_column_list
        all_column_list = self.col_names
        paired_column_list = list(itertools.product(categorical_column_list,all_column_list))
        result_paired_columns = []
        for element in paired_column_list:
            if element[0] != element[1]:
                result_paired_columns.append((element[0], element[1]))
        return result_paired_columns

    def get_correlated_numerical_columns(self, min_absolute_coeff = 0.3):
        """This method generates a list of paired columns of correlated numerical data

        :param min_absolute_coeff: Minimum absolute value of correlation coefficient, defaults to 0.3
        :type min_absolute_coeff: float, optional
        :return: List of paired columns of correlated numerical data
        :rtype: list
        """
        df_new = self.get_filtered_dataframe(include_type=[np.number])
        new_columns = list(df_new.columns)
        result_paired_columns = []
        product_list = list(itertools.product(new_columns,new_columns))
        for element in product_list:
            if element[0] != element[1]:
                try:
                    df_col1 = df_new[element[0]]
                    df_col2 = df_new[element[1]]
                    if abs(df_col1.corr(df_col2)) >= float(min_absolute_coeff):
                        result_paired_columns.append(element)
                except Exception as e:
                    print('Error while checking correlation coefficient comparison ', element, e)
        return result_paired_columns

    def plot_correlation_map(self, save=True, show=False):
        """This method generates correlation map

        :param save: 'True' if correlation map is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if correlation map is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        df_num = self.get_filtered_dataframe(include_type=np.number)
        df_cont = self.df[self.continuous_column_list]
        corr_matrix_num = df_num.corr()
        plot = sns.heatmap(corr_matrix_num, annot=True)
        self.save_or_show(plot.figure, 'correlation_map', 'corr_map_all_numerical_cols', save=save, show=show)
        corr_matrix_cont = df_cont.corr()
        plot = sns.heatmap(corr_matrix_cont, annot=True)
        self.save_or_show(plot.figure, 'correlation_map', 'corr_map_continuous_cols', save=save, show=show)

    def plot_scatter_plots(self,  save = True, show = False):
        """This method generates scatter plots

        :param save: 'True' if scatter plot is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if scatter plot is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        df_new = self.get_filtered_dataframe()
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())

        for col_pair in col_pairs:
            y = col_pair[0]
            x = col_pair[1]
            try:
                sns_plot = sns.scatterplot(x=x, y=y, data=self.df)
                self.save_or_show(sns_plot.figure, 'scatter', str(x)+'_'+str(y),x_label = x, y_label = y, save=save, show=show)
            except Exception as e:
                print('Cannot plot scatter plot for column pair',col_pair, e)

    def plot_horizontal_box_plot(self, save = True, show = False):
        """This method generates box plots

        :param save: 'True' if box plot is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if box plot is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        new_df = self.df
        cat_cols = self.non_continuous_col_list
        num_cols = self.numerical_column_list
        cont_cols = self.continuous_column_list
        for x_col in cont_cols:
            sns_plot_1 = sns.boxplot(x = x_col, data = self.df)
            self.save_or_show(sns_plot_1.figure, 'box_plot', str(x_col),x_label = x_col, save=save, show=show)
            for y_col in cat_cols:
                if y_col in num_cols:
                    new_df[y_col] = new_df[y_col].astype('category')
                sns_plot = sns.boxplot(x = x_col, y = y_col, data = new_df)
                self.save_or_show(sns_plot.figure, 'box_plot', str(x_col)+'_'+str(y_col),x_label = x_col, y_label = y_col, save=save, show=show)

    def plot_regression_marginals(self, save = True, show = False):
        """This method generates regression marginal plots

        :param save: 'True' if regression marginal plot is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if regression marginal plot is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        df_new = self.get_filtered_dataframe(include_type=[np.number])
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())

        for col_pair in col_pairs:
            y = col_pair[0]
            x = col_pair[1]
            try:
                sns_plot = sns.jointplot(x, y, data=df_new,kind="reg", truncate=False)
                self.save_or_show(sns_plot, 'regression_marginals', str(x)+'_'+str(y),x_label = x, y_label = y, save=save, show=show)
            except Exception as e:
                print('Cannot plot regression marginal plot for column pair',col_pair, e)

    def plot_scatter_plot_with_categorical(self, save = True, show = False):
        """This method generates categorical scatter plots

        :param save: 'True' if categorical scatter plot is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if categorical scatter plot is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        cat_cols = self.non_continuous_col_list
        num_cols = self.continuous_column_list
        for cat_col in cat_cols:
            for num_col in num_cols:
                sns_plot = sns.swarmplot(x=cat_col, y=num_col, data=self.df)
                self.save_or_show(sns_plot.figure, 'scatter_plot_categorical', str(cat_col)+'_'+str(num_col),x_label = cat_col, y_label = num_col, save=save, show=show)

    def plot_scatter_plot_matrix(self, hue_col_list=[], save=True, show=False):
        """This method generates scatter plot matrix

        :param hue_col_list: Columns to map plot aspects in different colors, defaults to []
        :type hue_col_list: list, optional
        :param save: 'True' if scatter plot matrix is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if scatter plot matrix is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        for col in self.categorical_column_list:
            sns_plot = sns.pairplot(self.df, x_vars=self.categorical_column_list, y_vars=self.numerical_column_list,hue = col, dropna=True)
            self.save_or_show(sns_plot, 'scatterplot_matrix', 'hue_'+str(col),x_label = col, save=save, show=show)

    def plot_paired_pointplots(self, save=True, show=False):
        """This method generates point plots

        :param save: 'True' if point plot is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if point plot is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        if self.target_column not in self.categorical_column_list:
            for column in self.categorical_column_list:
                try:
                    plot = sns.pointplot(x=column, y=self.target_column, data=self.df)
                    self.save_or_show(plot.figure, 'point_plot', column + "_" + self.target_column,x_label = column, y_label = self.target_column, save=save, show=show)
                except Exception as e:
                    print('Cannot plot pointplot for column ',column, e)
        else:
            print('Target column is not categorical')

    def plot_pie_chart(self,x = None, y = None, save = True, show = False, threshold = 10):
        """This method generates pie charts

        :param x: x variable, defaults to None
        :type x: NoneType, optional
        :param y: y variable, defaults to None
        :type y: NoneType, optional
        :param save: 'True' if pie chart is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if pie chart is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        :param threshold: Slice visibility threshold, defaults to 10
        :type threshold: int, optional
        """
        df_new = self.df[self.non_continuous_col_list]
        for col in df_new.columns:
            try:
                val_series = df_new[col].value_counts()
                val_name_list = list(val_series.keys())
                val_count_list = [ val_series[val_name] for val_name in val_name_list ]
                plot = plt.pie(val_count_list, labels=val_name_list)
                self.save_or_show(plt, 'piechart', str(col), x_label = col, save=save, show=show)
            except Exception as e:
                print('Cannot plot pie chart for column ',col, e)

    def plot_histogram(self, save=True, show=False):
        """This method generates histograms

        :param save: 'True' if histogram is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if histogram is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        df = self.get_filtered_dataframe(include_type=np.number)
        for column in df:
            try:
                values = list(df[column])
                plot = sns.distplot(values, bins='auto', kde=False)
                self.save_or_show(plot.figure, 'histogram', column,x_label = column, save=save, show=show)
            except Exception as e:
                print('Cannot plot Histogram ',e)

    def plot_line_chart(self, save=True, show=False):
        """This method generates line charts

        :param save: 'True' if line chart is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if line chart is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
         xs = []
         for col in self.col_names:
             if self.df[col].shape[0] == self.df[col].unique().shape[0]:
                       xs.append(col)
         for x in xs:
             res = []
             for i,j in zip(self.df[x], self.df.iloc[:,-1]):
                 res.append([i,j])
             res.sort()
             x1 = [x1[0] for x1 in res]
             y1 = [y1[1] for y1 in res]
             try:
                 plot1 = plt.plot(x1,y1)
                 self.save_or_show(plt, 'Line Chart', 'Line_Chart'+"_"+x,x_label = x, y_label = self.target_column ,save=save, show=show)
             except Exception as e:
               print('Cannot plot Line Chart',e)

    def plot_diagonal_correlation_matrix(self, save=True, show=False):
        """This method generates diagonal correlation matrix

        :param save: 'True' if diagonal correlation matrix is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if diagonal correlation matrix is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        corr = self.df.corr()
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        try:
            plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
            self.save_or_show(plot.figure, 'Diagonal_correlation_matrix', 'Diagonal_correlation_matrix', save=save, show=show)
        except Exception as e:
           print('Cannot plot Diagonal_correlation_matrix',e)

    def plot_stem_plots(self,  save = True, show = False ):
        """This method generates stem plots

        :param save: 'True' if stem plot is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if stem plot is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        df_new = self.get_filtered_dataframe(include_type=[np.number])
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())

        for col_pair in col_pairs:
            y = col_pair[0]
            x = col_pair[1]
            try:
                sns_plot = plt.stem(df_new[x], df_new[y],use_line_collection=True)
                self.save_or_show(plt, 'stem', str(x)+'_'+str(y), x_label = x, y_label = y, save=save, show=show)
            except Exception as e:
                print('Cannot plot stem plot for column pair',col_pair, e)

    def plot_kde(self, save=True, show=False):
        """This method generates Kernel Density Estimation(KDE) charts

        :param save: 'True' if KDE chart is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if KDE chart is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        col_names = self.numerical_column_list
        for i in range(len(col_names)):
            for j in range(i+1,len(col_names)):
                 try:
                    ax = sns.kdeplot((self.df[col_names[i]]), self.df[(col_names[j])])
                    self.save_or_show(ax.figure, 'KDE Chart', col_names[i] + "_"+ col_names[j], x_label = col_names[i], y_label = col_names[j],save=save, show=show)
                 except Exception as e:
                    print('Cannot plot kde',e)

    def plot_jitter_stripplot(self, save=True, show=False):
        """This method generates strip plots 

        :param save: 'True' if strip plot is to be saved as figure and 'False' otherwise, defaults to True
        :type save: bool, optional
        :param show: 'True' if strip plot is to be shown(displayed) and 'False' otherwise, defaults to False
        :type show: bool, optional
        """
        column_list = self.categorical_column_list
        if self.target_column in column_list:
            print('Ignoring as target column is in categorical_column_list')
            return
        y = self.df[self.target_column]
        if len(column_list) > 1:
            for col1, col2 in itertools.combinations(column_list, 2):
                if self.df[col2].nunique() > self.df[col1].nunique():
                    col1, col2 = col2, col1
                plot = sns.stripplot(x=self.df[col1], y=y, hue=self.df[col2])
                self.save_or_show(plot.figure, 'jitter_stripplot', col1+'_'+col2, x_label = col1, y_label = y, save=save, show=show)
        else:
            plot = sns.stripplot(x=list(self.df[column_list[0]]), y=y)
            self.save_or_show(plot.figure, 'jitter_stripplot', column_list[0], save=save, show=show)
