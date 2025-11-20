"""
Модуль парсингу та завантаження даних
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os


def parse_minfin_data(url='https://index.minfin.com.ua/ua/labour/wagemin/',
                      use_backup=True):
    """
    Парсинг даних прожиткового мінімуму з Minfin
    """
    print("\nПарсинг даних з Minfin.com.ua...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            df_list = pd.read_html(response.text)

            if df_list and len(df_list) > 0:
                df = df_list[0]
                parsed_data = _process_minfin_data(df)

                print(f"Успішно спарсено: {len(parsed_data)} записів")

                _save_parsed_to_csv(parsed_data, source='parsed_minfin_data')
                return parsed_data

    except Exception as e:
        print(f"Помилка парсингу: {e}")

    if use_backup:
        print("Використовуємо резервні реалістичні дані")
        return _generate_realistic_backup()

        _save_parsed_to_csv(backup_data, source='backup_data')

    raise Exception("Не вдалося отримати дані")


def _process_minfin_data(df):
    """Обробка спарсених даних"""
    df = df.dropna()
    df.columns = ['period', 'total', 'children_under_6', 'children_6_18',
                  'working_age', 'disabled']

    dates, values, years, months = [], [], [], []

    for idx, row in df.iterrows():
        period_str = str(row['period'])
        match = re.search(r'з\s+(\d{2})\.(\d{2})\.(\d{4})', period_str)

        if match:
            day, month, year = match.group(1), int(match.group(2)), int(match.group(3))

            try:
                date = pd.to_datetime(f"{day}.{month:02d}.{year}", format='%d.%m.%Y')
                dates.append(date)
                values.append(float(row['total']))
                years.append(year)
                months.append(month)
            except:
                continue

    df_processed = pd.DataFrame({
        'date': dates,
        'living_wage': values,
        'year': years,
        'month': months
    }).sort_values('date').reset_index(drop=True)

    return df_processed


def _generate_realistic_backup():
    """Генерація резервних реалістичних даних"""
    dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='QS')
    n = len(dates)

    base_value = 270
    years_passed = np.arange(n) / 4
    growth_rate = 0.08

    living_wage = base_value * np.exp(growth_rate * years_passed)
    noise = np.random.normal(0, 20, n)
    living_wage = np.round(living_wage + noise).astype(int)

    df = pd.DataFrame({
        'date': dates,
        'living_wage': living_wage,
        'year': dates.year.values,
        'month': dates.month.values
    })

    return df


def _save_parsed_to_csv(df, source='data'):
    """Збереження спарсених даних у CSV"""
    save_dir = 'parsed_data'
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{source}.csv"
    filepath = os.path.join(save_dir, filename)

    df.to_csv(filepath, index=False, encoding='utf-8-sig')

    file_size = os.path.getsize(filepath) / 1024
    print(f"Дані збережено: {filepath} ({file_size:.2f} KB)")


def load_excel_dataset(filepath):
    """
    Завантаження dataset
    """
    print(f"\nЗавантаження Excel файлу...")

    try:
        df = pd.read_excel(filepath, engine='openpyxl')

        print(f"Файл завантажено")
        print(f"Розмір: {df.shape[0]} рядків × {df.shape[1]} колонок")

        df = _clean_dataset(df)

        return df

    except Exception as e:
        print(f"Помилка завантаження: {e}")
        raise


def _clean_dataset(df):
    """Очищення та підготовка dataset"""

    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)

    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    df = df.drop_duplicates()

    if 'Order Date' in df.columns:
        df = df.sort_values('Order Date').reset_index(drop=True)

    print(f"Дані очищено: {df.shape[0]} рядків залишилось")

    return df


def prepare_timeseries_from_dataset(df, value_column, date_column='Order Date',
                                    aggregation='sum', freq='M'):
    """
    Підготовка часового ряду з dataset

    Parameters:
    -----------
    df : DataFrame
        Вхідні дані
    value_column : str
        Колонка зі значеннями
    date_column : str
        Колонка з датами
    aggregation : str
        Тип агрегації ('sum', 'mean', 'count')
    freq : str
        Частота ('D', 'W', 'M', 'Q', 'Y')
    """

    if date_column not in df.columns:
        raise ValueError(f"Колонка {date_column} не знайдена")

    if value_column not in df.columns:
        raise ValueError(f"Колонка {value_column} не знайдена")

    df_ts = df.set_index(date_column)[[value_column]]

    if aggregation == 'sum':
        df_ts = df_ts.resample(freq).sum()
    elif aggregation == 'mean':
        df_ts = df_ts.resample(freq).mean()
    elif aggregation == 'count':
        df_ts = df_ts.resample(freq).count()
    else:
        raise ValueError(f"Невідомий тип агрегації: {aggregation}")

    df_ts = df_ts.dropna()

    return df_ts