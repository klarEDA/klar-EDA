from os import makedirs
from os.path import join, exists
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from constants import VIZ_ROOT, NUNIQUE_THRESHOLD

class CSVVisualize:

    def __init__(self, input, target_col = None, index_column = None, exclude_columns = []):
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
        self.categorical_column_list = []
        self.populate_categorical_column_list()
        self.numerical_column_list = list(self.get_filtered_dataframe(include_type=np.number))
        temp_col_list = [num_col for num_col in self.numerical_column_list if self.df[num_col].nunique() < NUNIQUE_THRESHOLD]
        self.continuous_column_list = [x for x in self.numerical_column_list if x not in temp_col_list]
        self.non_continuous_col_list = self.categorical_column_list + temp_col_list
    
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

    def get_filtered_dataframe(self, include_type = [], exclude_type = []):
        if include_type or exclude_type:
            return self.df.select_dtypes(include = include_type, exclude = exclude_type)
        else:
            return self.df

    def populate_categorical_column_list(self):
        df = self.get_filtered_dataframe(exclude_type=np.number)
        if not self.categorical_column_list:
            for column in df:
                if df[column].nunique() <= NUNIQUE_THRESHOLD:
                    self.categorical_column_list.append(column)

    def get_categorical_numerical_columns_pairs(self):
        categorical_column_list = self.categorical_column_list
        all_column_list = self.col_names
        paired_column_list = list(itertools.product(categorical_column_list,all_column_list))
        result_paired_columns = []
        for element in paired_column_list:
            if element[0] != element[1]:
                result_paired_columns.append((element[0], element[1]))
        return result_paired_columns

    def get_correlated_numerical_columns(self, min_absolute_coeff = 0.3):

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
        df_num = self.get_filtered_dataframe(include_type=np.number)
        df_cont = self.df[self.continuous_column_list]
        corr_matrix_num = df_num.corr()
        plot = sns.heatmap(corr_matrix_num, annot=True)
        self.save_or_show(plot.figure, 'correlation_map', 'corr_map_all_numerical_cols', save=save, show=show)
        corr_matrix_cont = df_cont.corr()
        plot = sns.heatmap(corr_matrix_cont, annot=True)
        self.save_or_show(plot.figure, 'correlation_map', 'corr_map_continuous_cols', save=save, show=show)

    def get_numerical_column_list(self):
        pass

    def plot_numerical_feature_distribution(self):
        pass

    def plot_categorical_feature_distribution(self):
        pass

    def plot_scatter_plots(self,  save = True, show = False):

        df_new = self.get_filtered_dataframe()
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())

        for col_pair in col_pairs:
            y = col_pair[0]
            x = col_pair[1]
            try:
                sns_plot = sns.scatterplot(x=x, y=y, data=self.df)
                self.save_or_show(sns_plot.figure, 'scatter', str(x)+'_'+str(y), save=save, show=show)
            except Exception as e:
                print('Cannot plot scatter plot for column pair',col_pair, e)

    def plot_grid_plots_for_categorical_features(self, save_to_file = True):
        pass

    def plot_horizontal_box_plot(self, save = True, show = False):
        new_df = self.df
        cat_cols = self.non_continuous_col_list
        num_cols = self.numerical_column_list
        cont_cols = self.continuous_column_list
        for x_col in cont_cols:
            sns_plot_1 = sns.boxplot(x = x_col, data = self.df)
            self.save_or_show(sns_plot_1.figure, 'box_plot', str(x_col), save=save, show=show)
            for y_col in cat_cols:
                if y_col in num_cols:
                    new_df[y_col] = new_df[y_col].astype('category')
                sns_plot = sns.boxplot(x = x_col, y = y_col, data = new_df)
                self.save_or_show(sns_plot.figure, 'box_plot', str(x_col)+'_'+str(y_col), save=save, show=show)

    def plot_pdp(self):
        pass

    def plot_regression_marginals(self, save = True, show = False):

        df_new = self.get_filtered_dataframe(include_type=[np.number])
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())

        for col_pair in col_pairs:
            y = col_pair[0]
            x = col_pair[1]
            try:
                sns_plot = sns.jointplot(x, y, data=df_new,kind="reg", truncate=False)
                self.save_or_show(sns_plot, 'regression_marginals', str(x)+'_'+str(y), save=save, show=show)
            except Exception as e:
                print('Cannot plot regression marginal plot for column pair',col_pair, e)

    def plot_scatter_plot_with_categorical(self, save = True, show = False):
        cat_cols = self.non_continuous_col_list
        num_cols = self.continuous_column_list
        for cat_col in cat_cols:
            for num_col in num_cols:
                sns_plot = sns.swarmplot(x=cat_col, y=num_col, data=self.df)
                self.save_or_show(sns_plot.figure, 'scatter_plot_categorical', str(cat_col)+'_'+str(num_col), save=save, show=show)

    def plot_scatter_plot_matrix(self, hue_col_list=[], save=True, show=False):
        #if len(hue_col_list)>0:
        #    cat_col_list = hue_col_list
        #else:
        #    cat_col_list = list(self.get_filtered_dataframe(exclude_type=[np.number]).columns)
        #sns_plot = sns.pairplot(df_new)
        #self.save_or_show(sns_plot.figure, 'scatterplot_matrix', 'no_hue', save=save, show=show)
        for col in self.categorical_column_list:
            #print('plotting',col)
            sns_plot = sns.pairplot(self.df, x_vars=self.categorical_column_list, y_vars=self.numerical_column_list,hue = col, dropna=True)
            self.save_or_show(sns_plot, 'scatterplot_matrix', 'hue_'+str(col), save=save, show=show)

    def plot_paired_pointplots(self, save=True, show=False):
        if self.target_column not in self.categorical_column_list:
            for column in self.categorical_column_list:
                try:
                    plot = sns.pointplot(x=column, y=self.target_column, data=self.df)
                    self.save_or_show(plot.figure, 'point_plot', column + "_" + self.target_column, save=save, show=show)
                except Exception as e:
                    print('Cannot plot pointplot for column ',column, e)
        else:
            print('Target column is not categorical')

    def plot_pie_chart(self,x = None, y = None, save = True, show = False, threshold = 10):
        df_new = self.df[self.non_continuous_col_list]
        for col in df_new.columns:
            try:
                val_series = df_new[col].value_counts()
                val_name_list = list(val_series.keys())
                val_count_list = [ val_series[val_name] for val_name in val_name_list ]
                plot = plt.pie(val_count_list, labels=val_name_list)
                self.save_or_show(plt, 'piechart', str(col), save=save, show=show)
            except Exception as e:
                print('Cannot plot pie chart for column ',col, e)

    def plot_histogram(self, save=True, show=False):
        df = self.get_filtered_dataframe(include_type=np.number)
        for column in df:
            try:
                values = list(df[column])
                plot = sns.distplot(values, bins='auto', kde=False)
                self.save_or_show(plot.figure, 'histogram', column, save=save, show=show)
            except Exception as e:
                print('Cannot plot Histogram ',e)

    def plot_line_chart(self, save=True, show=False):
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
                 self.save_or_show(plt, 'Line Chart', 'Line_Chart'+"_"+x,save=save, show=show)
             except Exception as e:
               print('Cannot plot Line Chart',e)

    def plot_diagonal_correlation_matrix(self, save=True, show=False):
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

        df_new = self.get_filtered_dataframe(include_type=[np.number])
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())
        
        for col_pair in col_pairs:
            y = col_pair[0]
            x = col_pair[1]
            try:
                sns_plot = plt.stem(df_new[x], df_new[y],use_line_collection=True)
                self.save_or_show(plt, 'stem', str(x)+'_'+str(y), save=save, show=show)
            except Exception as e:
                print('Cannot plot stem plot for column pair',col_pair, e)
    
    def plot_kde(self, save=True, show=False):
        col_names = self.numerical_column_list
        for i in range(len(col_names)):
            for j in range(i+1,len(col_names)):
                 try:
                    ax = sns.kdeplot((self.df[col_names[i]]), self.df[(col_names[j])])
                    self.save_or_show(ax.figure, 'KDE Chart', col_names[i] + "_"+ col_names[j],save=save, show=show)
                 except Exception as e:
                    print('Cannot plot kde',e)
                    
    def plot_jitter_stripplot(self, save=True, show=False):
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
                self.save_or_show(plot.figure, 'jitter_stripplot', col1+'_'+col2, save=save, show=show)
        else:
            plot = sns.stripplot(x=list(self.df[column_list[0]]), y=y)
            self.save_or_show(plot.figure, 'jitter_stripplot', column_list[0], save=save, show=show) 
