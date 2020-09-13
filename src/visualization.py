from klar_eda.visualize.csv_visualize import CSVVisualize
from klar_eda.visualize.image_visualize import ImageDataVisualize

def visualize_csv(csv, target_col=None, index_column=None, exclude_columns = []):
    visualizer = CSVVisualize(csv, target_col=target_col, index_column=index_column, exclude_columns=exclude_columns)
    visualizer.plot_scatter_plot_matrix()
    visualizer.plot_correlation_map()
    visualizer.plot_histogram()
    visualizer.plot_scatter_plots()
    visualizer.plot_regression_marginals()
    visualizer.plot_pie_chart()
    visualizer.plot_stem_plots()
    visualizer.plot_scatter_plot_with_categorical()
    visualizer.plot_horizontal_box_plot()
    visualizer.plot_line_chart()
    visualizer.plot_line_chart()
    visualizer.plot_diagonal_correlation_matrix()
    visualizer.plot_kde()

def visualize_images(data, labels, boxes=None, save=True, show=False):
    visualizer = ImageDataVisualize(data, labels, boxes=boxes)
    visualizer.aspect_ratio_histogram(save=save, show=show)
    visualizer.area_vs_category(save=save, show=show)
    visualizer.mean_images(save=save, show=show)
    visualizer.eigen_images(save=save, show=show)
    visualizer.num_images_by_category(save=save, show=show)
    visualizer.std_vs_mean(save=save, show=show)
    visualizer.t_sne(save=save, show=show)
