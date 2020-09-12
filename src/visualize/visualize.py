from csv_visualize import CSVVisualize

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
    file_path = "" #add path to your test data
    visualize_csv(file_path)
    
test_csv_visualization()