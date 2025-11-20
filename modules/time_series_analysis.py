"""
Модуль аналізу часових рядів:
- Декомпозиція
- Виявлення властивостей (стаціонарність, фрактальність)
- Кореляційний аналіз
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf


def decompose_time_series(data, period=12, model='additive'):
    """
    Декомпозиція часового ряду на тренд, сезонність та залишки

    Parameters:
    -----------
    data : array-like
        Часовий ряд
    period : int
        Період сезонності
    model : str
        'additive' або 'multiplicative'

    Returns:
    --------
    dict : Компоненти декомпозиції
    """
    print(f"Декомпозиція (період={period}, модель={model})")

    if len(data) < 2 * period:
        period = max(2, len(data) // 4)
        print(f"Період скориговано до {period}")

    try:
        decomposition = seasonal_decompose(data, model=model, period=period,
                                           extrapolate_trend='freq')

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        trend_strength = 1 - np.var(residual) / np.var(trend + residual)
        seasonal_strength = 1 - np.var(residual) / np.var(seasonal + residual)

        result = {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength,
            'model': model,
            'period': period
        }

        print(f"Сила тренду: {trend_strength:.3f}")
        print(f"Сила сезонності: {seasonal_strength:.3f}")

        return result

    except Exception as e:
        print(f"Помилка декомпозиції: {e}")
        return None


def analyze_properties(data):
    """
    Комплексний аналіз властивостей часового ряду

    Returns:
    --------
    dict : Властивості часового ряду
    """
    print("Аналіз властивостей...")

    properties = {}

    properties['mean'] = np.mean(data)
    properties['std'] = np.std(data)
    properties['cv'] = properties['std'] / properties['mean'] if properties['mean'] != 0 else 0
    properties['min'] = np.min(data)
    properties['max'] = np.max(data)
    properties['range'] = properties['max'] - properties['min']

    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
    properties['trend_slope'] = slope
    properties['trend_r2'] = r_value ** 2
    properties['trend_pvalue'] = p_value

    trend_type = "зростаючий" if slope > 0 else "спадний" if slope < 0 else "відсутній"
    print(f"Тренд: {trend_type} (нахил={slope:.4f})")

    try:
        adf_result = adfuller(data, autolag='AIC')
        properties['adf_statistic'] = adf_result[0]
        properties['adf_pvalue'] = adf_result[1]
        properties['adf_critical_1%'] = adf_result[4]['1%']
        properties['adf_critical_5%'] = adf_result[4]['5%']

        is_stationary = adf_result[1] < 0.05
        properties['is_stationary_adf'] = is_stationary

        print(f"ADF тест: {'стаціонарний' if is_stationary else 'нестаціонарний'} (p={adf_result[1]:.4f})")
    except:
        properties['is_stationary_adf'] = None

    try:
        kpss_result = kpss(data, regression='c', nlags='auto')
        properties['kpss_statistic'] = kpss_result[0]
        properties['kpss_pvalue'] = kpss_result[1]
        properties['kpss_critical_5%'] = kpss_result[3]['5%']

        is_stationary_kpss = kpss_result[1] > 0.05
        properties['is_stationary_kpss'] = is_stationary_kpss

        print(f"KPSS тест: {'стаціонарний' if is_stationary_kpss else 'нестаціонарний'} (p={kpss_result[1]:.4f})")
    except:
        properties['is_stationary_kpss'] = None

    hurst = calculate_hurst_exponent(data)
    properties['hurst_exponent'] = hurst

    if hurst < 0.5:
        hurst_type = "антиперсистентний (mean-reverting)"
    elif hurst > 0.5:
        hurst_type = "персистентний (trending)"
    else:
        hurst_type = "випадкове блукання"

    print(f"Показник Херста: H={hurst:.3f} ({hurst_type})")

    seasonality_info = detect_seasonality(data)
    properties['seasonality'] = seasonality_info

    detrended = data - (slope * x + intercept)
    properties['noise'] = {
        'mean': np.mean(detrended),
        'std': np.std(detrended),
        'distribution': 'normal'
    }

    properties['trend'] = {
        'type': 'linear',
        'slope': slope,
        'intercept': intercept
    }

    properties['stationarity'] = {
        'adf': properties.get('is_stationary_adf'),
        'kpss': properties.get('is_stationary_kpss')
    }

    return properties


def calculate_hurst_exponent(data):
    """
    Розрахунок показника Херста (R/S аналіз)
    """
    n = len(data)

    if n < 20:
        return 0.5

    lags = range(2, min(n // 2, 100))
    tau = []

    for lag in lags:
        n_blocks = n // lag
        rs_values = []

        for i in range(n_blocks):
            block = data[i * lag:(i + 1) * lag]
            mean_block = np.mean(block)
            cumsum = np.cumsum(block - mean_block)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(block, ddof=1)

            if S > 0:
                rs_values.append(R / S)

        if rs_values:
            tau.append(np.mean(rs_values))
        else:
            tau.append(0)

    lags = np.array(list(lags))
    tau = np.array(tau)

    mask = tau > 0
    lags = lags[mask]
    tau = tau[mask]

    if len(lags) < 2:
        return 0.5

    hurst, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    return hurst


def detect_seasonality(data, max_period=None):
    """
    Виявлення сезонності через спектральний аналіз
    """
    if max_period is None:
        max_period = len(data) // 2

    freqs, power = signal.periodogram(data)

    if len(power) > 1:
        peak_idx = np.argmax(power[1:]) + 1

        if freqs[peak_idx] > 0:
            dominant_period = int(1 / freqs[peak_idx])
        else:
            dominant_period = None
    else:
        dominant_period = None

    return {
        'detected': dominant_period is not None,
        'period': dominant_period,
        'strength': np.max(power[1:]) / np.sum(power) if len(power) > 1 else 0
    }


def calculate_correlations(data, max_lag=40):
    """
    Розрахунок автокореляції та часткової автокореляції
    """
    print(f"Кореляційний аналіз (max_lag={max_lag})")

    max_lag = min(max_lag, len(data) // 2 - 1)

    acf_values = acf(data, nlags=max_lag, fft=True)
    pacf_values = pacf(data, nlags=max_lag, method='ywm')

    cov_matrix = np.cov(data[:-1], data[1:])
    correlation = cov_matrix[0, 1] / (np.std(data[:-1]) * np.std(data[1:]))

    result = {
        'acf': acf_values,
        'pacf': pacf_values,
        'lag1_correlation': correlation,
        'max_lag': max_lag
    }

    print(f"ACF/PACF розраховано для {max_lag} лагів")
    print(f"Кореляція lag-1: {correlation:.3f}")

    return result


def calculate_distance_metrics(ts1, ts2):
    """
    Розрахунок метрик подібності між двома часовими рядами
    """
    min_len = min(len(ts1), len(ts2))
    ts1 = ts1[:min_len]
    ts2 = ts2[:min_len]

    euclidean = np.sqrt(np.sum((ts1 - ts2) ** 2))
    euclidean_norm = euclidean / min_len
    cosine_sim = np.dot(ts1, ts2) / (np.linalg.norm(ts1) * np.linalg.norm(ts2))
    correlation, _ = stats.pearsonr(ts1, ts2)
    dtw_dist = calculate_dtw_distance(ts1, ts2)

    return {
        'euclidean': euclidean,
        'euclidean_normalized': euclidean_norm,
        'cosine_similarity': cosine_sim,
        'correlation': correlation,
        'dtw': dtw_dist
    }


def calculate_dtw_distance(ts1, ts2, window=10):
    """
    Dynamic Time Warping відстань
    """
    n, m = len(ts1), len(ts2)

    if n > 1000 or m > 1000:
        return np.sqrt(np.sum((ts1[:min(n, m)] - ts2[:min(n, m)]) ** 2))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window)):
            cost = abs(ts1[i - 1] - ts2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m]