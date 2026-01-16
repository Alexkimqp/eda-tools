"""
Exploratory Data Analysis (EDA) utilities module.

This module provides functions for performing basic exploratory data analysis
on pandas DataFrames, including statistics, missing values, correlations, and more.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Union

def basic_information(dataset: pd.DataFrame) -> None:
    """Prints basic exploratory information about a pandas DataFrame.

    The function validates the input dataset and outputs:
    - Dataset shape (rows and columns)
    - Total number of missing values
    - Total number of duplicated rows
    - Count of data types
    - First five rows of the dataset

    Args:
        dataset (pd.DataFrame): Input dataset for basic exploration.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        ValueError: If the DataFrame is empty.

    Returns:
        None: This function prints information to the console and does not return a value.
    """

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")

    if dataset.empty:
        raise ValueError("Dataset is empty.")
  
    row, col = dataset.shape
    print(f"❗Shape: rows {row} columns {col}")
    print("=" * 40)

    total_missing = dataset.isnull().sum().sum()
  
    if total_missing == 0:
        print("There are no missing values in the dataset.")
    else:
        print(f"❗Total missing: {total_missing}")

    print("=" * 40)

    dublicate = dataset.duplicated().sum()
  
    if dublicate == 0:
        print("There are no dublicates in the dataset.")
    else:
        print(f"❗Total dublicates: {dublicate}")

    print("=" * 40)

    data_types = dataset.dtypes.value_counts()
    print("❗Dtypes (count):")
    print(data_types)

    print("=" * 40)
    print(dataset.head(5))


def basic_eda(df: pd.DataFrame) -> None:
    """Perform basic exploratory data analysis on a DataFrame.
    
    This function prints a comprehensive EDA report including:
    - DataFrame shape (rows and columns)
    - Missing values by column
    - Duplicate rows count and percentage
    - Column names
    - Data types distribution
    - Unique values count per column
    
    Args:
        df: Input pandas DataFrame to analyze.
        
    Raises:
        ValueError: If DataFrame is empty.
        TypeError: If input is not a pandas DataFrame.
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, None]})
        >>> basic_eda(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    print("══════════ 📊 EDA REPORT ══════════")
    
    # Размер
    row, col = df.shape
    print(f"❗Shape: {row:,} rows × {col:,} cols")
    print("=" * 40)

    # Пропуски по колонкам
    missing_by_col = df.isna().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    missing_total = int(missing_by_col.sum())

    print("❗Missing")
    if len(missing_by_col) == 0:
        print("No missing values found")
    else:
        for col_name in missing_by_col.index:
            print(f"{col_name} - {missing_by_col[col_name]}")
    
    print(f"❗Total missing: {missing_total:,}")
    print("=" * 40)

    # Дубликаты
    duplicate = df.duplicated().sum()
    duplicate_pct = ((duplicate / len(df)) * 100).round(2) if len(df) > 0 else 0
    print(f"❗Duplicate: {duplicate:,}")
    print(f"❗Percentage of duplicates: {duplicate_pct}%")
    print("=" * 40)

    # Список имен колонок
    cols_preview = list(df.columns)
    print(f"❗Columns ({len(cols_preview)}): {cols_preview}")
    print("=" * 40)

    # Типы данных
    data_types = df.dtypes.value_counts()
    print("❗Dtypes (count):")
    print(data_types)
    print("=" * 40)

    # Количество уникальных значений по колонкам
    unique_counts = df.nunique()
    print("❗Unique values:")
    print(unique_counts)
    print("=" * 40)


def top_corr(
    df: pd.DataFrame,
    n: int = 10,
    method: str = 'pearson',
    heatmap: bool = False,
    figsize: tuple[int, int] = (10, 8)
) -> Optional[pd.Series]:
    """Find top N feature correlations in a DataFrame.
    
    This function calculates correlations between numeric features and returns
    the top N pairs by absolute correlation value. Optionally displays a heatmap.
    
    Args:
        df: Input pandas DataFrame with numeric columns.
        n: Number of top correlations to return. Defaults to 10.
        method: Correlation method to use. Options: 'pearson', 'kendall', 'spearman'.
            Defaults to 'pearson'.
        heatmap: If True, displays a correlation heatmap. Defaults to False.
        figsize: Figure size for the heatmap as (width, height) tuple. Defaults to (10, 8).
        
    Returns:
        pandas Series with top N correlations, or None if no numeric columns found.
        Index contains pairs of feature names, values are correlation coefficients.
        
    Raises:
        ValueError: If method is not one of the supported correlation methods.
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [1, 1, 1, 1, 1]})
        >>> top_corr(df, n=5, heatmap=True)
    """
    if method not in ['pearson', 'kendall', 'spearman']:
        raise ValueError(f"Method must be 'pearson', 'kendall', or 'spearman', got '{method}'")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("❗Warning: Need at least 2 numeric columns for correlation analysis")
        return None
    
    corr_matrix = df[numeric_cols].corr(method=method)
    np.fill_diagonal(corr_matrix.values, np.nan)
    corr_unstack = corr_matrix.abs().unstack()
    corr_unstack = corr_unstack[
        corr_unstack.index.get_level_values(0) < corr_unstack.index.get_level_values(1)
    ]
    top_n = corr_unstack.sort_values(ascending=False).head(n)

    print(f"❗Top {n} features by correlation (method: {method})")
    print("=" * 50)
    print(top_n)
    
    if heatmap:
        corr_matrix_for_heatmap = df[numeric_cols].corr(method=method)
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix_for_heatmap, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=0.5)
        plt.title(f'Correlation Heatmap ({method.capitalize()})')
        plt.tight_layout()
        plt.show()
    
    return top_n


def corr_by_target(
    df: pd.DataFrame,
    target: str,
    method: str = 'pearson',
    n: int = 10
) -> Optional[pd.Series]:
    """Find top N features correlated with a target variable.
    
    This function calculates correlations between numeric features and a specified
    target column, returning the top N features by absolute correlation value.
    
    Args:
        df: Input pandas DataFrame with numeric columns.
        target: Name of the target column to correlate with.
        method: Correlation method to use. Options: 'pearson', 'kendall', 'spearman'.
            Defaults to 'pearson'.
        n: Number of top correlations to return. Defaults to 10.
        
    Returns:
        pandas Series with top N correlations with target, or None if target not found
        or not numeric. Index contains feature names, values are correlation coefficients.
        
    Raises:
        KeyError: If target column does not exist in DataFrame.
        ValueError: If method is not one of the supported correlation methods.
        TypeError: If target column is not numeric.
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4, 5],
        ...     'B': [2, 4, 6, 8, 10],
        ...     'target': [3, 6, 9, 12, 15]
        ... })
        >>> corr_by_target(df, target='target', n=5)
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")
    
    if method not in ['pearson', 'kendall', 'spearman']:
        raise ValueError(f"Method must be 'pearson', 'kendall', or 'spearman', got '{method}'")
    
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError(f"Target column '{target}' must be numeric")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target not in numeric_cols:
        raise TypeError(f"Target column '{target}' is not numeric")
    
    if len(numeric_cols) < 2:
        print("❗Warning: Need at least 2 numeric columns for correlation analysis")
        return None
    
    corr_matrix = df[numeric_cols].corr(method=method)
    corr_target = (
        corr_matrix[target]
        .abs()
        .drop(target)
        .sort_values(ascending=False)
        .head(n)
    )

    print(f"❗Top {n} features by correlation with '{target}' (method: {method})")
    print("=" * 50)
    print(corr_target)
    
    return corr_target
