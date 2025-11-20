"""
Пакет модулів для лабораторної роботи №4
Аналіз та кластеризація часових рядів
"""

from .data_parser import (
    parse_minfin_data,
    load_excel_dataset,
    prepare_timeseries_from_dataset
)

from .time_series_analysis import (
    decompose_time_series,
    analyze_properties,
    calculate_correlations,
    calculate_distance_metrics
)

from .clustering import (
    perform_clustering,
    optimal_clusters,
    cluster_time_series_segments
)

from .visualization import (
    create_visualizations,
    plot_decomposition,
    plot_properties,
    plot_correlations,
    plot_clustering
)

from .synthetic_generator import (
    generate_synthetic_timeseries,
    add_anomalies,
    generate_with_hurst,
    validate_synthetic_vs_real
)

__version__ = '1.0.0'
__author__ = 'Дяченко Данііл'

__all__ = [
    'parse_minfin_data',
    'load_excel_dataset',
    'prepare_timeseries_from_dataset',
    'decompose_time_series',
    'analyze_properties',
    'calculate_correlations',
    'calculate_distance_metrics',
    'perform_clustering',
    'optimal_clusters',
    'cluster_time_series_segments',
    'create_visualizations',
    'plot_decomposition',
    'plot_properties',
    'plot_correlations',
    'plot_clustering',
    'generate_synthetic_timeseries',
    'add_anomalies',
    'generate_with_hurst',
    'validate_synthetic_vs_real'
]