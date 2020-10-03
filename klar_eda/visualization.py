from .visualize.csv_visualize import CSVVisualize
from .visualize.image_visualize import ImageDataVisualize

def visualize_csv(csv, target_col=None, index_column=None, exclude_columns=[], save=True, show=False):
    """Generates various visualization charts using heuristic techniques 
    for the csv data and save into a folder 'Plots'.

    :param csv: Either pandas Dataframe ( with column names as row 0 ) OR path to csv file
    :type csv: pandas.Dataframe/ string
    :param target_col: Name of the target column, defaults to last column in the dataframe.
    :type target_col: string, optional
    :param index_column: List of column names which contain indexes/ do not contribute at all, defaults to None
    :type index_column: list of string, optional
    :param exclude_columns: List of columns to be excluded from considering for visualization, defaults to []
    :type exclude_columns: list, optional
    :param save: Save the results to directory, defaults to True
    :type save: bool, optional
    :param show: Preview the results, defaults to False
    :type show: bool, optional
    """
    visualizer = CSVVisualize(csv, target_col=target_col, index_column=index_column, exclude_columns=exclude_columns)
    visualizer.plot_scatter_plot_matrix(save=save,show=show)
    visualizer.plot_correlation_map(save=save,show=show)
    visualizer.plot_histogram(save=save,show=show)
    visualizer.plot_scatter_plots(save=save,show=show)
    visualizer.plot_regression_marginals(save=save,show=show)
    visualizer.plot_pie_chart(save=save,show=show)
    visualizer.plot_stem_plots(save=save,show=show)
    visualizer.plot_scatter_plot_with_categorical(save=save,show=show)
    visualizer.plot_horizontal_box_plot(save=save,show=show)
    visualizer.plot_line_chart(save=save,show=show)
    visualizer.plot_diagonal_correlation_matrix(save=save,show=show)
    visualizer.plot_kde(save=save,show=show)
    print('CSV Visualization completed successfully!')

def visualize_images(data, labels, boxes=None, save=True, show=False):
    """Generates various visualization charts for given image data and 
    saves into a directory named 'Plots'.

    :param data: No description available
    :type data: [type]
    :param labels: No description available
    :type labels: [type]
    :param boxes: No description available, defaults to None
    :type boxes: [type], optional
    :param save: Save the results to directory, defaults to True
    :type save: bool, optional
    :param show: Preview the results, defaults to False
    :type show: bool, optional
    """
    visualizer = ImageDataVisualize(data, labels, boxes=boxes)
    visualizer.aspect_ratio_histogram(save=save, show=show)
    visualizer.area_vs_category(save=save, show=show)
    visualizer.mean_images(save=save, show=show)
    visualizer.eigen_images(save=save, show=show)
    visualizer.num_images_by_category(save=save, show=show)
    visualizer.std_vs_mean(save=save, show=show)
    visualizer.t_sne(save=save, show=show)
    print('Image Visualization completed successfully!')