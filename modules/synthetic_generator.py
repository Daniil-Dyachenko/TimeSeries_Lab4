"""
Модуль генерування синтетичних часових рядів
з заданими властивостями
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_timeseries(n, trend_params, seasonality_params,
                                  noise_params, stationarity, start_date=None):
    """
    Генерування синтетичного часового ряду з заданими властивостями

    Parameters:
    -----------
    n : int
        Кількість точок
    trend_params : dict
        Параметри тренду
    seasonality_params : dict
        Параметри сезонності
    noise_params : dict
        Параметри шуму
    stationarity : dict
        Інформація про стаціонарність
    start_date : datetime
        Початкова дата

    Returns:
    --------
    dict : Синтетичні дані
    """
    print("\nГенерування синтетичних даних...")

    if start_date is None:
        start_date = datetime(2000, 1, 1)

    dates = pd.date_range(start=start_date, periods=n, freq='QS')

    trend = generate_trend(n, trend_params)
    print(f"Тренд: {trend_params['type']} (slope={trend_params['slope']:.4f})")

    seasonal = generate_seasonality(n, seasonality_params)

    if seasonality_params.get('detected'):
        print(f"Сезонність: період={seasonality_params.get('period', 'N/A')}")
    else:
        print(f"Сезонність: відсутня")

    noise = generate_noise(n, noise_params)
    print(f"Шум: σ={noise_params['std']:.2f}")

    if stationarity.get('adf') and stationarity.get('kpss'):
        values = trend * 0.3 + seasonal + noise
        print(f"Стаціонарний ряд: тренд зменшено")
    else:
        values = trend + seasonal + noise
        print(f"Нестаціонарний ряд: повний тренд")

    if 'mean' in trend_params:
        target_mean = trend_params.get('intercept', 0)
        target_std = noise_params['std']

        current_mean = np.mean(values)
        current_std = np.std(values)

        if current_std > 0:
            values = (values - current_mean) / current_std * target_std + target_mean

    result = {
        'values': values,
        'dates': dates,
        'trend': trend,
        'seasonal': seasonal,
        'noise': noise,
        'n': n
    }

    return result


def generate_trend(n, params):
    """
    Генерування тренду
    """
    x = np.arange(n)

    trend_type = params.get('type', 'linear')

    if trend_type == 'linear':
        slope = params.get('slope', 0)
        intercept = params.get('intercept', 0)
        trend = slope * x + intercept

    elif trend_type == 'polynomial':
        coefficients = params.get('coefficients', [0, 0, 0])
        trend = np.polyval(coefficients, x)

    elif trend_type == 'exponential':
        rate = params.get('rate', 0.01)
        base = params.get('base', 100)
        trend = base * np.exp(rate * x / n)

    elif trend_type == 'logarithmic':
        scale = params.get('scale', 10)
        trend = scale * np.log(x + 1)

    else:
        trend = np.zeros(n)

    return trend


def generate_seasonality(n, params):
    """
    Генерування сезонної компоненти
    """
    if not params.get('detected', False):
        return np.zeros(n)

    period = params.get('period', 12)
    strength = params.get('strength', 0.1)

    n_harmonics = 2

    seasonal = np.zeros(n)

    for k in range(1, n_harmonics + 1):
        amplitude = strength / k

        phase = np.random.uniform(0, 2 * np.pi)

        seasonal += amplitude * np.sin(2 * np.pi * k * np.arange(n) / period + phase)

    if np.std(seasonal) > 0:
        seasonal = seasonal / np.std(seasonal) * strength

    return seasonal


def generate_noise(n, params):
    """
    Генерування шумової компоненти
    """
    distribution = params.get('distribution', 'normal')
    mean = params.get('mean', 0)
    std = params.get('std', 1)

    if distribution == 'normal':
        noise = np.random.normal(mean, std, n)

    elif distribution == 'uniform':
        a = mean - std * np.sqrt(3)
        b = mean + std * np.sqrt(3)
        noise = np.random.uniform(a, b, n)

    elif distribution == 'exponential':
        scale = std
        noise = np.random.exponential(scale, n) + mean - scale

    else:
        noise = np.random.normal(mean, std, n)

    return noise


def add_anomalies(data, anomaly_ratio=0.05, anomaly_magnitude=5):
    """
    Додавання аномалій до згенерованих синтезованих даних
    """
    n = len(data)
    n_anomalies = int(n * anomaly_ratio)

    data_with_anomalies = data.copy()

    anomaly_indices = np.random.choice(n, n_anomalies, replace=False)

    std = np.std(data)

    for idx in anomaly_indices:
        sign = np.random.choice([-1, 1])
        data_with_anomalies[idx] += sign * anomaly_magnitude * std

    return data_with_anomalies, anomaly_indices


def generate_with_hurst(n, hurst_exponent, mean=0, std=1):
    """
    Генерування ряду з заданим показником Херста
    """

    if abs(hurst_exponent - 0.5) < 0.05:
        return np.cumsum(np.random.normal(0, std, n)) + mean

    alpha = 2 * hurst_exponent - 1

    data = np.zeros(n)
    data[0] = np.random.normal(mean, std)

    for i in range(1, n):
        data[i] = alpha * data[i - 1] + (1 - alpha) * mean + np.random.normal(0, std * (1 - alpha ** 2) ** 0.5)

    return data


def generate_multiple_timeseries(n_series, n_points, base_params, variation=0.2):
    """
    Генерування набору схожих часових рядів з варіаціями

    Parameters:
    -----------
    n_series : int
        Кількість рядів
    n_points : int
        Кількість точок в кожному ряді
    base_params : dict
        Базові параметри
    variation : float
        Рівень варіації параметрів (0-1)

    Returns:
    --------
    list : Список часових рядів
    """
    series_list = []

    for i in range(n_series):
        trend_params = base_params['trend'].copy()
        trend_params['slope'] *= np.random.uniform(1 - variation, 1 + variation)

        seasonality_params = base_params['seasonality'].copy()
        if seasonality_params.get('detected'):
            seasonality_params['strength'] *= np.random.uniform(1 - variation, 1 + variation)

        noise_params = base_params['noise'].copy()
        noise_params['std'] *= np.random.uniform(1 - variation, 1 + variation)

        ts = generate_synthetic_timeseries(
            n=n_points,
            trend_params=trend_params,
            seasonality_params=seasonality_params,
            noise_params=noise_params,
            stationarity=base_params['stationarity']
        )

        series_list.append(ts['values'])

    return series_list


def validate_synthetic_vs_real(real_data, synthetic_data):
    """
    Валідація синтетичних даних порівняно з реальними

    Returns:
    --------
    dict : Метрики порівняння
    """
    from scipy import stats

    t_stat, t_pvalue = stats.ttest_ind(real_data, synthetic_data)

    f_stat = np.var(real_data) / np.var(synthetic_data)
    f_pvalue = stats.f.sf(f_stat, len(real_data) - 1, len(synthetic_data) - 1)

    ks_stat, ks_pvalue = stats.ks_2samp(real_data, synthetic_data)

    mean_diff = abs(np.mean(real_data) - np.mean(synthetic_data)) / np.mean(real_data) * 100
    std_diff = abs(np.std(real_data) - np.std(synthetic_data)) / np.std(real_data) * 100

    validation = {
        't_test': {'statistic': t_stat, 'pvalue': t_pvalue,
                   'similar': t_pvalue > 0.05},
        'f_test': {'statistic': f_stat, 'pvalue': f_pvalue,
                   'similar': f_pvalue > 0.05},
        'ks_test': {'statistic': ks_stat, 'pvalue': ks_pvalue,
                    'similar': ks_pvalue > 0.05},
        'mean_diff_%': mean_diff,
        'std_diff_%': std_diff,
        'overall_similar': (t_pvalue > 0.05 and ks_pvalue > 0.05)
    }

    print("\nВалідація синтетичних даних:")
    print(f"t-test (середні): p={t_pvalue:.4f} {'✓' if t_pvalue > 0.05 else '✗'}")
    print(f"KS-test (розподіл): p={ks_pvalue:.4f} {'✓' if ks_pvalue > 0.05 else '✗'}")
    print(f"Відхилення середнього: {mean_diff:.2f}%")
    print(f"Відхилення СКВ: {std_diff:.2f}%")

    return validation