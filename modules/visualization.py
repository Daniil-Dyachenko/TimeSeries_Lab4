"""
Модуль візуалізації результатів аналізу
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy import stats as scipy_stats
import os


def create_visualizations(data, dates, decomposition, properties,
                          correlations, clustering, save_dir):
    """
    Створення всіх візуалізацій
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Створення візуалізацій...")

    plot_decomposition(data, dates, decomposition, save_dir)
    plot_properties(data, dates, properties, save_dir)
    plot_correlations(correlations, save_dir)
    plot_clustering(data, dates, clustering, save_dir)
    plot_summary(data, dates, decomposition, properties, save_dir)

    print(f"Візуалізації збережено в {save_dir}/")


def plot_decomposition(data, dates, decomposition, save_dir):
    """
    Графік декомпозиції
    """
    if decomposition is None:
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    axes[0].plot(dates, data, 'b-', linewidth=1.5)
    axes[0].set_title('Оригінальний часовий ряд', fontweight='bold')
    axes[0].set_ylabel('Значення')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, decomposition['trend'], 'g-', linewidth=2)
    axes[1].set_title(f'Тренд (сила: {decomposition["trend_strength"]:.3f})',
                      fontweight='bold')
    axes[1].set_ylabel('Тренд')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, decomposition['seasonal'], 'orange', linewidth=1.5)
    axes[2].set_title(f'Сезонність (сила: {decomposition["seasonal_strength"]:.3f}, період: {decomposition["period"]})',
                      fontweight='bold')
    axes[2].set_ylabel('Сезонність')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(dates, decomposition['residual'], 'r-', linewidth=1, alpha=0.7)
    axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[3].set_title('Залишки (шум)', fontweight='bold')
    axes[3].set_ylabel('Залишки')
    axes[3].set_xlabel('Дата')
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(f'ДЕКОМПОЗИЦІЯ ЧАСОВОГО РЯДУ ({decomposition["model"]})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_decomposition.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_properties(data, dates, properties, save_dir):
    """
    Графік властивостей
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, data, 'b-', alpha=0.6, label='Дані')

    x = np.arange(len(data))
    trend_line = properties['trend_slope'] * x + properties['trend']['intercept']
    ax1.plot(dates, trend_line, 'r--', linewidth=2, label=f'Тренд (slope={properties["trend_slope"]:.4f})')

    ax1.set_title('Часовий ряд з трендом', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(properties['mean'], color='red', linestyle='--', linewidth=2, label=f'μ={properties["mean"]:.2f}')
    ax2.set_title('Розподіл значень', fontweight='bold')
    ax2.set_xlabel('Значення')
    ax2.set_ylabel('Частота')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.boxplot(data, vert=True)
    ax3.set_title('Box Plot', fontweight='bold')
    ax3.set_ylabel('Значення')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    scipy_stats.probplot(data, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (нормальність)', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    adf_status = "Стаціонарний" if properties.get('is_stationary_adf') else "Нестаціонарний"
    kpss_status = "Стаціонарний" if properties.get('is_stationary_kpss') else "Нестаціонарний"

    plt.suptitle('ВЛАСТИВОСТІ ЧАСОВОГО РЯДУ', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/02_properties.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlations(correlations, save_dir):
    """
    Графік кореляцій
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    acf = correlations['acf']
    pacf = correlations['pacf']
    lags = range(len(acf))

    ax1.stem(lags, acf, basefmt=' ')
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.axhline(y=1.96 / np.sqrt(len(acf)), color='red', linestyle='--', linewidth=1)
    ax1.axhline(y=-1.96 / np.sqrt(len(acf)), color='red', linestyle='--', linewidth=1)
    ax1.set_title('Автокореляційна функція (ACF)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Лаг')
    ax1.set_ylabel('Кореляція')
    ax1.grid(True, alpha=0.3)

    ax2.stem(lags, pacf, basefmt=' ')
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axhline(y=1.96 / np.sqrt(len(pacf)), color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=-1.96 / np.sqrt(len(pacf)), color='red', linestyle='--', linewidth=1)
    ax2.set_title('Часткова автокореляційна функція (PACF)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Лаг')
    ax2.set_ylabel('Часткова кореляція')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'КОРЕЛЯЦІЙНИЙ АНАЛІЗ (lag-1: {correlations["lag1_correlation"]:.3f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/03_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_clustering(data, dates, clustering, save_dir):
    """
    Графік кластеризації
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    if 'kmeans' in clustering:
        ax1 = fig.add_subplot(gs[0, 0])
        kmeans = clustering['kmeans']
        labels = kmeans['labels']

        window_size = clustering['window_size']
        full_labels = np.repeat(labels, window_size)[:len(data)]

        for cluster_id in range(kmeans['n_clusters']):
            mask = full_labels == cluster_id
            ax1.scatter(np.array(dates)[mask], data[mask],
                        label=f'Кластер {cluster_id}', alpha=0.6, s=20)

        ax1.set_title(f'K-means (Silhouette: {kmeans["silhouette"]:.3f})', fontweight='bold')
        ax1.set_xlabel('Дата')
        ax1.set_ylabel('Значення')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    if 'hierarchical' in clustering:
        ax2 = fig.add_subplot(gs[0, 1])
        hier = clustering['hierarchical']
        labels = hier['labels']

        window_size = clustering['window_size']
        full_labels = np.repeat(labels, window_size)[:len(data)]

        for cluster_id in range(hier['n_clusters']):
            mask = full_labels == cluster_id
            ax2.scatter(np.array(dates)[mask], data[mask],
                        label=f'Кластер {cluster_id}', alpha=0.6, s=20)

        ax2.set_title(f'Hierarchical (Silhouette: {hier["silhouette"]:.3f})', fontweight='bold')
        ax2.set_xlabel('Дата')
        ax2.set_ylabel('Значення')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, :])
        dendrogram(hier['linkage_matrix'], ax=ax3, color_threshold=0, above_threshold_color='black')
        ax3.set_title('Дендрограма', fontweight='bold')
        ax3.set_xlabel('Індекс вікна')
        ax3.set_ylabel('Відстань')
        ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle('КЛАСТЕРИЗАЦІЯ', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/04_clustering.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary(data, dates, decomposition, properties, save_dir):
    """
    Загальний підсумковий графік
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, data, 'b-', linewidth=1.5, alpha=0.7)
    ax1.set_title('Часовий ряд', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Значення', fontsize=11)
    ax1.grid(True, alpha=0.3)

    info_text = f"n={len(data)} | μ={properties['mean']:.2f} | σ={properties['std']:.2f} | H={properties.get('hurst_exponent', 0):.3f}"
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if decomposition:
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(dates, decomposition['trend'], 'g-', linewidth=2)
        ax2.set_title(f'Тренд (сила: {decomposition["trend_strength"]:.3f})', fontweight='bold')
        ax2.set_ylabel('Тренд')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(dates, decomposition['seasonal'], 'orange', linewidth=1.5)
        ax3.set_title(f'Сезонність (період: {decomposition["period"]})', fontweight='bold')
        ax3.set_ylabel('Сезонність')
        ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black', density=True)
    from scipy.stats import norm
    mu, std = properties['mean'], properties['std']
    xmin, xmax = ax4.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax4.plot(x, p, 'r-', linewidth=2, label='Норм. розподіл')
    ax4.set_title('Розподіл', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    plt.suptitle('ЗАГАЛЬНИЙ АНАЛІЗ', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/05_summary.png", dpi=300, bbox_inches='tight')
    plt.close()