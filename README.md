# EDA Tools

A collection of Python functions for performing Exploratory Data Analysis (EDA) on pandas DataFrames.

## Features

- **Basic EDA Report**: Comprehensive overview of your dataset including shape, missing values, duplicates, data types, and unique values
- **Top Correlations**: Find the top N feature correlations in your dataset
- **Target Correlations**: Analyze correlations between features and a target variable

## Installation

### From GitHub (after publishing)

```bash
pip install git+https://github.com/Alexkimqp/eda-tools.git
```

### Local Installation

```bash
git clone https://github.com/Alexkimqp/eda-tools.git
cd eda-tools
pip install -e .
```

## Quick Start

```python
import pandas as pd
from my_eda import basic_eda, top_corr, corr_by_target

# Load your data
df = pd.read_csv('your_data.csv')

# Basic EDA report
basic_eda(df)

# Top 10 correlations
top_corr(df, n=10, heatmap=True)

# Correlations with target variable
corr_by_target(df, target='target_column', n=10)
```

## Functions

### `get_basic_information(dataset: pd.DataFrame) -> dict`

Collects basic exploratory information about a pandas DataFrame and returns it as a dictionary.

**Parameters:**
- `dataset`: Input pandas DataFrame to analyze

**Returns:**
- None: This function prints information to the console and does not return a value.


### `basic_eda(df: pd.DataFrame) -> None`

Performs basic exploratory data analysis and prints a comprehensive report.

**Parameters:**
- `df`: pandas DataFrame to analyze

**Example:**
```python
basic_eda(df)
```

### `top_corr(df: pd.DataFrame, n: int = 10, method: str = 'pearson', heatmap: bool = False, figsize: tuple[int, int] = (10, 8)) -> pd.Series`

Finds top N feature correlations in a DataFrame.

**Parameters:**
- `df`: Input pandas DataFrame with numeric columns
- `n`: Number of top correlations to return (default: 10)
- `method`: Correlation method - 'pearson', 'kendall', or 'spearman' (default: 'pearson')
- `heatmap`: If True, displays a correlation heatmap (default: False)
- `figsize`: Figure size for the heatmap as (width, height) tuple (default: (10, 8))

**Returns:**
- pandas Series with top N correlations

**Example:**
```python
top_corr(df, n=5, method='spearman', heatmap=True)
```

### `corr_by_target(df: pd.DataFrame, target: str, method: str = 'pearson', n: int = 10) -> pd.Series`

Finds top N features correlated with a target variable.

**Parameters:**
- `df`: Input pandas DataFrame with numeric columns
- `target`: Name of the target column to correlate with
- `method`: Correlation method - 'pearson', 'kendall', or 'spearman' (default: 'pearson')
- `n`: Number of top correlations to return (default: 10)

**Returns:**
- pandas Series with top N correlations with target

**Example:**
```python
corr_by_target(df, target='price', n=10)
```

## Requirements

- Python >= 3.7
- pandas >= 1.0.0
- numpy >= 1.18.0
- matplotlib >= 3.0.0
- seaborn >= 0.10.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

