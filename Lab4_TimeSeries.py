"""
Лабораторна робота №4: АНАЛІЗ ТА КЛАСТЕРИЗАЦІЯ TIME SERIES
Завдання ІІI рівня – максимально 15 балів

Реалізувати Групи вимог_1 та довести ефективність запропонованих рішень шляхом
реалізації Групи вимог_2

Група вимог_1:
1. Декомпозиція, виявлення властивостей, візуалізація, кластеризація,
   кореляційний аналіз Time Series
2. Опис виявлених закономірностей
3. Генерування синтетичних Time Series з виявленими властивостями
4. Модульна структура для подальшого використання

Група вимог_2:
Довести ефективність на DataSet

"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

from modules.data_parser import parse_minfin_data, load_excel_dataset
from modules.time_series_analysis import (
    decompose_time_series,
    analyze_properties,
    calculate_correlations
)
from modules.clustering import perform_clustering
from modules.visualization import create_visualizations
from modules.synthetic_generator import generate_synthetic_timeseries

warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (14, 8)

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False


class TimeSeriesLab4:
    """
    Головний клас
    """

    def __init__(self, output_dir='Lab4_Results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/graphs", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)

        self.real_data = None
        self.synthetic_data = None
        self.dataset_data = None
        self.analysis_results = {}

    def load_real_data(self):
        """Завантаження реальних даних (парсинг Minfin)"""
        print("\n" + "-" * 80)
        print("ЗАВАНТАЖЕННЯ РЕАЛЬНИХ ДАНИХ")

        self.real_data = parse_minfin_data()
        print(f"\nЗавантажено {len(self.real_data)} записів")
        print(f"Період: {self.real_data['date'].min().date()} → {self.real_data['date'].max().date()}")
        print(f"Діапазон значень: {self.real_data['living_wage'].min():.0f} - {self.real_data['living_wage'].max():.0f} грн")

        return self.real_data

    def analyze_real_data(self):
        """Поглиблений аналіз реальних даних"""
        print("\n" + "-" * 80)
        print("АНАЛІЗ РЕАЛЬНИХ ДАНИХ")

        if self.real_data is None:
            raise ValueError("Спочатку завантажте реальні дані!")

        ts_data = self.real_data['living_wage'].values
        dates = self.real_data['date'].values

        print("\nДекомпозиція часового ряду...")
        decomposition = decompose_time_series(ts_data, period=4)
        self.analysis_results['real_decomposition'] = decomposition

        properties = analyze_properties(ts_data)
        self.analysis_results['real_properties'] = properties

        correlations = calculate_correlations(ts_data, max_lag=12)
        self.analysis_results['real_correlations'] = correlations

        clustering = perform_clustering(ts_data, dates, n_clusters=3)
        self.analysis_results['real_clustering'] = clustering

        create_visualizations(
            ts_data, dates, decomposition, properties,
            correlations, clustering,
            f"{self.output_dir}/graphs/real_data"
        )

        return self.analysis_results

    def generate_synthetic(self):
        """Генерування синтетичних даних з виявленими властивостями"""
        print("\n" + "-" * 80)
        print("ГЕНЕРУВАННЯ СИНТЕТИЧНИХ ДАНИХ НА ОСНОВІ ВИЯВЛЕНИХ ВЛАСТИВОСТЕЙ")

        if 'real_properties' not in self.analysis_results:
            raise ValueError("Спочатку проаналізуйте реальні дані!")

        properties = self.analysis_results['real_properties']
        n = len(self.real_data)

        print(f"\nГенерування {n} точок на основі виявлених властивостей...")

        self.synthetic_data = generate_synthetic_timeseries(
            n=n,
            trend_params=properties['trend'],
            seasonality_params=properties['seasonality'],
            noise_params=properties['noise'],
            stationarity=properties['stationarity']
        )

        print(f"\nСинтетичні дані згенеровано")
        print(f"Середнє: {self.synthetic_data['values'].mean():.2f}")
        print(f"СКВ: {self.synthetic_data['values'].std():.2f}")

        df_synthetic = pd.DataFrame({
            'date': self.synthetic_data['dates'],
            'value': self.synthetic_data['values']
        })
        df_synthetic.to_csv(f"{self.output_dir}/data/synthetic_data.csv", index=False)

        return self.synthetic_data

    def analyze_synthetic_data(self):
        """Аналіз синтетичних даних"""
        print("-" * 80)
        print("АНАЛІЗ СИНТЕТИЧНИХ ДАНИХ")

        if self.synthetic_data is None:
            raise ValueError("Спочатку згенерруйте синтетичні дані!")

        ts_data = self.synthetic_data['values']
        dates = self.synthetic_data['dates']

        decomposition = decompose_time_series(ts_data, period=4)
        self.analysis_results['synthetic_decomposition'] = decomposition

        properties = analyze_properties(ts_data)
        self.analysis_results['synthetic_properties'] = properties

        correlations = calculate_correlations(ts_data, max_lag=12)
        self.analysis_results['synthetic_correlations'] = correlations

        clustering = perform_clustering(ts_data, dates, n_clusters=3)
        self.analysis_results['synthetic_clustering'] = clustering

        create_visualizations(
            ts_data, dates, decomposition, properties,
            correlations, clustering,
            f"{self.output_dir}/graphs/synthetic_data"
        )

        return self.analysis_results

    def compare_real_vs_synthetic(self):
        """Порівняння реальних та синтетичних даних"""
        print("\n" + "-" * 80)
        print("ПОРІВНЯННЯ РЕАЛЬНИХ ТА СИНТЕТИЧНИХ ДАНИХ")

        real_props = self.analysis_results['real_properties']
        synth_props = self.analysis_results['synthetic_properties']

        comparison = {
            'Характеристика': [],
            'Реальні дані': [],
            'Синтетичні дані': [],
            'Відхилення (%)': []
        }

        metrics = [
            ('Середнє', 'mean'),
            ('СКВ', 'std'),
            ('Коеф. варіації', 'cv'),
            ('Тренд (нахил)', 'trend_slope'),
            ('Стаціонарність (ADF p-value)', 'adf_pvalue'),
            ('Показник Херста', 'hurst_exponent')
        ]

        for name, key in metrics:
            real_val = real_props.get(key, 0)
            synth_val = synth_props.get(key, 0)

            if real_val != 0:
                deviation = abs((synth_val - real_val) / real_val * 100)
            else:
                deviation = 0

            comparison['Характеристика'].append(name)
            comparison['Реальні дані'].append(f"{real_val:.4f}")
            comparison['Синтетичні дані'].append(f"{synth_val:.4f}")
            comparison['Відхилення (%)'].append(f"{deviation:.2f}%")

        df_comparison = pd.DataFrame(comparison)
        print("\n" + df_comparison.to_string(index=False))

        self._plot_comparison()

        return df_comparison

    def analyze_dataset(self, filepath):
        """Аналіз наданого DataSet"""
        print("\n" + "-" * 80)
        print("АНАЛІЗ DATASETУ ")

        print(f"\nЗавантаження: {filepath}")
        self.dataset_data = load_excel_dataset(filepath)

        print(f"\nЗавантажено {len(self.dataset_data)} записів")
        print(f"Колонки: {list(self.dataset_data.columns)}")

        time_series_columns = ['Sales', 'Discount', 'Profit', 'Quantity']

        for col in time_series_columns:
            if col in self.dataset_data.columns:
                print(f"\n--- Аналіз: {col} ---")
                ts_data = self.dataset_data[col].values
                dates = pd.to_datetime(
                    self.dataset_data['Order Date']) if 'Order Date' in self.dataset_data.columns else np.arange(
                    len(ts_data))

                decomposition = decompose_time_series(ts_data, period=12)
                properties = analyze_properties(ts_data)
                correlations = calculate_correlations(ts_data, max_lag=20)
                clustering = perform_clustering(ts_data, dates, n_clusters=4)

                create_visualizations(
                    ts_data, dates, decomposition, properties,
                    correlations, clustering,
                    f"{self.output_dir}/graphs/dataset_{col}"
                )

                self.analysis_results[f'dataset_{col}'] = {
                    'decomposition': decomposition,
                    'properties': properties,
                    'correlations': correlations,
                    'clustering': clustering
                }

        self._analyze_cross_correlation()

        return self.dataset_data

    def _analyze_cross_correlation(self):
        """Крос-кореляційний аналіз між рядами dataset"""
        print("\nКрос-кореляція між рядами")

        numeric_cols = self.dataset_data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.dataset_data[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1)
        plt.title('Кореляційна матриця Dataset', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/graphs/dataset_correlation_matrix.png", dpi=300)
        plt.close()

        print("Кореляційна матриця збережена")

    def _plot_comparison(self):
        """Візуальне порівняння реальних та синтетичних даних"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        real_ts = self.real_data['living_wage'].values
        synth_ts = self.synthetic_data['values']

        axes[0, 0].plot(real_ts, 'b-', label='Реальні дані', alpha=0.7)
        axes[0, 0].plot(synth_ts, 'r--', label='Синтетичні дані', alpha=0.7)
        axes[0, 0].set_title('Порівняння часових рядів')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(real_ts, bins=30, alpha=0.6, label='Реальні', color='blue', density=True)
        axes[0, 1].hist(synth_ts, bins=30, alpha=0.6, label='Синтетичні', color='red', density=True)
        axes[0, 1].set_title('Порівняння розподілів')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        from statsmodels.tsa.stattools import acf
        real_acf = acf(real_ts, nlags=20)
        synth_acf = acf(synth_ts, nlags=20)

        axes[1, 0].plot(real_acf, 'b-o', label='Реальні', markersize=4)
        axes[1, 0].plot(synth_acf, 'r--s', label='Синтетичні', markersize=4)
        axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 0].set_title('Автокореляційні функції')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        data_to_plot = [real_ts, synth_ts]
        axes[1, 1].boxplot(data_to_plot, labels=['Реальні', 'Синтетичні'])
        axes[1, 1].set_title('Box Plot порівняння')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('ПОРІВНЯННЯ: Реальні vs Синтетичні дані',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/graphs/comparison_real_vs_synthetic.png", dpi=300)
        plt.close()

        print("Графік порівняння збережено")

    def run_full_analysis(self, dataset_path=None):
        print("-" * 80)
        print("ЛАБОРАТОРНА РОБОТА №4")
        print("АНАЛІЗ ТА КЛАСТЕРИЗАЦІЯ TIME SERIES")
        print("-" * 80)

        try:
            self.load_real_data()
            self.analyze_real_data()
            self.generate_synthetic()
            self.analyze_synthetic_data()
            self.compare_real_vs_synthetic()

            dataset_path = 'Data_Set_11.xlsx'

            print("\n" + "-" * 80)
            print("ЗАВАНТАЖЕННЯ DATASET")
            print(f"\nФайл: {dataset_path}")

            if os.path.exists(dataset_path):
                print(f"Файл знайдено")
                file_size = os.path.getsize(dataset_path) / 1024  # KB
                print(f"Розмір: {file_size:.1f} KB")
                self.analyze_dataset(dataset_path)
            else:
                print(f"Файл '{dataset_path}' не знайдено")
                print(f"Переконайтесь, що файл знаходиться в папці проекту")

            print("\n" + "-" * 80)
            print("ВИКОНАННЯ ЗАВЕРШЕНО УСПІШНО")
            print("-" * 80)
            print(f"\nРезультати збережено в: {self.output_dir}/")
            print(f"Графіки: {self.output_dir}/graphs/")
            print(f"Дані: {self.output_dir}/data/")

        except Exception as e:
            print(f"\n ПОМИЛКА: {e}")
            raise

def main():
    lab = TimeSeriesLab4(output_dir='Lab4_Results')
    lab.run_full_analysis()

if __name__ == "__main__":
    main()