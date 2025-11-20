"""
Модуль кластеризації часових рядів:
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import euclidean


def perform_clustering(data, dates, n_clusters=3, methods=['kmeans', 'hierarchical']):
    """
    Кластеризація часового ряду

    Parameters:
    -----------
    data : array-like
        Часовий ряд
    dates : array-like
        Дати (для візуалізації)
    n_clusters : int
        Кількість кластерів
    methods : list
        Методи кластеризації

    Returns:
    --------
    dict : Результати кластеризації
    """
    print(f"Кластеризація (k={n_clusters})")

    window_size = max(3, len(data) // 20)
    features = create_windows(data, window_size)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    results = {}

    if 'kmeans' in methods:
        kmeans_result = apply_kmeans(features_scaled, n_clusters)
        results['kmeans'] = kmeans_result
        print(f"K-means: Silhouette={kmeans_result['silhouette']:.3f}")

    if 'hierarchical' in methods:
        hier_result = apply_hierarchical(features_scaled, n_clusters)
        results['hierarchical'] = hier_result
        print(f"Hierarchical: Silhouette={hier_result['silhouette']:.3f}")

    results['features'] = features
    results['features_scaled'] = features_scaled
    results['window_size'] = window_size
    results['dates'] = dates
    results['original_data'] = data

    return results


def create_windows(data, window_size, step=1):
    """
    Створення ковзаючих вікон для features
    """
    windows = []

    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]

        features = [
            np.mean(window),
            np.std(window),
            np.max(window) - np.min(window),
            np.median(window),
            (window[-1] - window[0]) / window_size if window_size > 1 else 0  # slope
        ]

        windows.append(features)

    return np.array(windows)


def apply_kmeans(features, n_clusters):
    """
    K-means кластеризація
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    inertia = kmeans.inertia_

    centers = kmeans.cluster_centers_

    return {
        'labels': labels,
        'centers': centers,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'inertia': inertia,
        'n_clusters': n_clusters
    }


def apply_hierarchical(features, n_clusters):
    """
    Ієрархічна кластеризація
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(features)

    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)

    linkage_matrix = linkage(features, method='ward')

    return {
        'labels': labels,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'linkage_matrix': linkage_matrix,
        'n_clusters': n_clusters
    }


def optimal_clusters(features, max_k=10):
    """
    Визначення оптимальної кількості кластерів
    """
    inertias = []
    silhouettes = []
    k_range = range(2, min(max_k + 1, len(features)))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(features, kmeans.labels_))

    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes
    }


def cluster_time_series_segments(data, n_clusters=3):
    """
    Кластеризація сегментів часового ряду
    (розбиття ряду на рівні частини та кластеризація)
    """
    n = len(data)
    segment_size = n // 10

    if segment_size < 3:
        segment_size = n // 3

    segments = []
    segment_features = []

    for i in range(0, n - segment_size + 1, segment_size):
        segment = data[i:i + segment_size]
        segments.append(segment)

        features = [
            np.mean(segment),
            np.std(segment),
            np.max(segment),
            np.min(segment),
            (segment[-1] - segment[0]) / len(segment),
            np.median(np.abs(np.diff(segment)))
        ]

        segment_features.append(features)

    segment_features = np.array(segment_features)

    scaler = StandardScaler()
    segment_features_scaled = scaler.fit_transform(segment_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(segment_features_scaled)

    return {
        'segments': segments,
        'labels': labels,
        'features': segment_features,
        'segment_size': segment_size
    }


def calculate_cluster_statistics(data, labels):
    """
    Розрахунок статистик для кожного кластеру
    """
    unique_labels = np.unique(labels)
    stats = {}

    for label in unique_labels:
        mask = labels == label
        cluster_data = data[mask]

        stats[f'cluster_{label}'] = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(data) * 100,
            'mean': np.mean(cluster_data),
            'std': np.std(cluster_data),
            'min': np.min(cluster_data),
            'max': np.max(cluster_data)
        }

    return stats


def dtw_clustering(time_series_list, n_clusters=3):
    """
    Кластеризація на основі DTW відстані
    """
    n_series = len(time_series_list)

    distance_matrix = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(i + 1, n_series):
            dist = simple_dtw(time_series_list[i], time_series_list[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )

    labels = hierarchical.fit_predict(distance_matrix)

    return {
        'labels': labels,
        'distance_matrix': distance_matrix,
        'n_clusters': n_clusters
    }


def simple_dtw(ts1, ts2):
    """
    Спрощена DTW відстань
    """
    n, m = len(ts1), len(ts2)

    if n > 500 or m > 500:
        min_len = min(n, m)
        return euclidean(ts1[:min_len], ts2[:min_len])

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(ts1[i - 1] - ts2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m]