# -*- coding: utf-8 -*-
"""
process_bank_churn.py

Модуль для попередньої обробки даних банківського відтоку клієнтів.
Містить функції для підготовки тренувальних, валідаційних та нових даних.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# ── Допоміжні функції ─────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Визначає числові та категоріальні колонки у датафреймі.

    Parameters
    ----------
    df : pd.DataFrame
        Вхідний датафрейм з ознаками (без цільової колонки).
    target_col : str
        Назва цільової колонки для виключення з ознак.

    Returns
    -------
    Tuple[List[str], List[str]]
        Кортеж (numeric_cols, categorical_cols).
    """
    input_df = df.drop(columns=[target_col], errors='ignore')
    numeric_cols = input_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols


def scale_numeric(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler.
    Навчає scaler на тренувальних даних і застосовує до обох наборів.

    Parameters
    ----------
    train_df : pd.DataFrame
        Тренувальний датафрейм.
    val_df : pd.DataFrame
        Валідаційний датафрейм.
    numeric_cols : List[str]
        Список числових колонок для масштабування.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]
        Масштабовані train, val датафрейми та навчений scaler.
    """
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return train_df, val_df, scaler


def encode_categorical(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Кодує категоріальні ознаки за допомогою OneHotEncoder.
    Навчає encoder на тренувальних даних і застосовує до обох наборів.

    Parameters
    ----------
    train_df : pd.DataFrame
        Тренувальний датафрейм.
    val_df : pd.DataFrame
        Валідаційний датафрейм.
    categorical_cols : List[str]
        Список категоріальних колонок для кодування.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]
        Закодовані train, val датафрейми, навчений encoder та список нових колонок.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_df = train_df.copy()
    val_df = val_df.copy()

    encoder.fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])

    return train_df, val_df, encoder, encoded_cols


# ── Основна функція препроцесингу ─────────────────────────────────────────────

def preprocess_data(
    bank_df: pd.DataFrame,
    target_col: str = 'Exited',
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric_features: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           List[str], MinMaxScaler, OneHotEncoder]:
    """
    Виконує повну попередню обробку банківських даних.

    Кроки обробки:
    1. Видалення непотрібних колонок (Surname та інші).
    2. Розбиття на тренувальний та валідаційний набори зі стратифікацією.
    3. One-Hot кодування категоріальних ознак.
    4. (Опціонально) Масштабування числових ознак через MinMaxScaler.

    Parameters
    ----------
    bank_df : pd.DataFrame
        Вхідний датафрейм з сирими даними.
    target_col : str
        Назва цільової колонки. За замовчуванням 'Exited'.
    drop_cols : Optional[List[str]]
        Колонки для видалення. За замовчуванням ['id', 'CustomerId', 'Surname'].
    test_size : float
        Частка валідаційного набору. За замовчуванням 0.2.
    random_state : int
        Seed для відтворюваності. За замовчуванням 42.
    scale_numeric_features : bool
        Якщо True — масштабує числові ознаки. За замовчуванням True.
        Для дерев рішень можна передати False.

    Returns
    -------
    Tuple containing:
        X_train : pd.DataFrame — тренувальні вхідні дані
        train_targets : pd.Series — тренувальні цільові значення
        X_val : pd.DataFrame — валідаційні вхідні дані
        val_targets : pd.Series — валідаційні цільові значення
        input_cols : List[str] — перелік колонок, що використовуються як вхід
        scaler : MinMaxScaler — навчений scaler (або None якщо scale_numeric_features=False)
        encoder : OneHotEncoder — навчений encoder
    """
    if drop_cols is None:
        drop_cols = ['id', 'CustomerId', 'Surname']

    # 1. Видаляємо непотрібні колонки
    df = bank_df.drop(columns=[c for c in drop_cols if c in bank_df.columns])

    # 2. Визначаємо input_cols
    input_cols = [col for col in df.columns if col != target_col]

    # 3. Розбиття на train/val зі стратифікацією
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col]
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col]

    # 4. Визначаємо типи колонок
    numeric_cols, categorical_cols = get_feature_columns(
        train_inputs.assign(**{target_col: 0}), target_col
    )
    numeric_cols = train_inputs.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include=['object']).columns.tolist()

    # 5. Масштабування числових (опціонально)
    scaler = None
    if scale_numeric_features and numeric_cols:
        train_inputs, val_inputs, scaler = scale_numeric(
            train_inputs, val_inputs, numeric_cols
        )

    # 6. Кодування категоріальних
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical(
        train_inputs, val_inputs, categorical_cols
    )

    # 7. Фінальні X з числовими + закодованими колонками
    final_cols = numeric_cols + encoded_cols
    X_train = train_inputs[final_cols]
    X_val = val_inputs[final_cols]

    return X_train, train_targets, X_val, val_targets, final_cols, scaler, encoder


# ── Функція для обробки нових даних ──────────────────────────────────────────

def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler] = None,
    drop_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Обробляє нові дані (наприклад test.csv) з використанням
    вже навченого scaler та encoder.

    Parameters
    ----------
    new_df : pd.DataFrame
        Новий датафрейм для обробки (наприклад test.csv).
    input_cols : List[str]
        Перелік колонок, що використовувались при навчанні моделі.
    encoder : OneHotEncoder
        Навчений на тренувальних даних encoder.
    scaler : Optional[MinMaxScaler]
        Навчений на тренувальних даних scaler.
        Якщо None — масштабування не виконується.
    drop_cols : Optional[List[str]]
        Колонки для видалення. За замовчуванням ['id', 'CustomerId', 'Surname'].

    Returns
    -------
    pd.DataFrame
        Оброблений датафрейм готовий для передбачення моделлю.
    """
    if drop_cols is None:
        drop_cols = ['id', 'CustomerId', 'Surname']

    df = new_df.copy()
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Визначаємо типи колонок
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Масштабування числових
    if scaler is not None and numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Кодування категоріальних
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    df[encoded_cols] = encoder.transform(df[categorical_cols])

    # Повертаємо лише потрібні колонки
    final_cols = [col for col in input_cols if col in df.columns]
    return df[final_cols]


# ── Приклад використання ──────────────────────────────────────────────────────

if __name__ == '__main__':
    # Для дерев (без масштабування)
    # X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder = \
    #     preprocess_data(bank_df, scale_numeric_features=False)

    # Для логістичної регресії (з масштабуванням)
    # X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder = \
    #     preprocess_data(bank_df, scale_numeric_features=True)

    # Обробка нових даних
    # X_test = preprocess_new_data(test_df, input_cols, encoder, scaler)
    pass
