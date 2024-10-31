import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import pickle



####################################################################################################################

def plot_categorical_countplots(df, n_cols=4, figname=None):
    """
    Plot count plots for all categorical features in the dataset, arranged in a grid.

    Parameters:
    - df (pd.DataFrame): The cleaned dataset containing categorical columns to plot.
    - n_cols (int): The number of columns per row in the grid for plotting. Default is 4.
    - figname (str or None): Full path to save the figure. If None, the plot will be displayed.

    Returns:
    - None. The function displays or saves the count plots in a grid layout.
    """
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    n_rows = int(len(categorical_features) / n_cols) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4), sharey=False)
    axes = axes.flatten()

    for i, col in enumerate(categorical_features):
        sns.countplot(x=col, data=df, ax=axes[i])
        axes[i].set_title(f'Count plot of {col}')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if figname:
        plt.savefig(figname, format='png')
    else:
        plt.show()

####################################################################################################################

def plot_numerical_histograms(df, log_scale=True, bins=50, figname=None):
    """
    Plot histograms for all numerical columns in the dataset, with an option to apply a log scale on the y-axis.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - log_scale (bool): Whether to apply a log scale to the y-axis. Default is True.
    - bins (int): The number of bins for the histograms. Default is 50.
    - figname (str or None): Full path to save the figure. If None, the plot will be displayed.

    Returns:
    - None. The function displays or saves histograms of numerical columns.
    """
    numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_vars].hist(bins=bins, figsize=(16, 12), log=log_scale)
    plt.tight_layout()

    if figname:
        plt.savefig(figname, format='png')
    else:
        plt.show()

####################################################################################################################

def plot_target_distribution(df, target='Incurred_capped_97.5p', bins=50, figname=None):
    """
    Plot the distribution of the target variable using a histogram with KDE (Kernel Density Estimate).

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the target variable.
    - target (str): The name of the target variable to plot.
    - bins (int): The number of bins for the histogram. Default is 50.
    - figname (str or None): Full path to save the figure. If None, the plot will be displayed.

    Returns:
    - None. The function displays or saves the distribution plot.
    """
    plt.figure(figsize=(9, 5))
    plt.title(f'Distribution of {target}')
    sns.histplot(df[target], color='g', kde=True, bins=bins)
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.tight_layout()

    if figname:
        plt.savefig(figname, format='png')
    else:
        plt.show()

####################################################################################################################

def plot_numeric_boxplots(df, numeric_vars, num_cols=3, figsize=(16, 12), figname=None):
    """
    Plots boxplots for numeric variables in the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numeric variables to plot.
    - numeric_vars (list): A list of numeric variable names to plot boxplots for.
    - num_cols (int): The number of columns to display per row of subplots (default is 3).
    - figsize (tuple): The size of the figure for the plots (default is (16, 12)).
    - figname (str or None): Full path to save the figure. If None, the plot will be displayed.

    Returns:
    - None. The function displays or saves the boxplots.
    """
    num_vars = len(numeric_vars)
    num_rows = (num_vars + num_cols - 1) // num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, var in enumerate(numeric_vars):
        sns.boxplot(y=df[var], ax=axes[i])
        axes[i].set_title(f'Boxplot of {var}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Value')

    for j in range(len(numeric_vars), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if figname:
        plt.savefig(figname, format='png')
    else:
        plt.show()

####################################################################################################################

def plot_correlation_heatmap(df, numeric_vars, target_var='Incurred_capped_97.5p', verbose=False, figname=None):
    """
    Plot a correlation heatmap for numeric variables, improving readability for crowded plots.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numeric variables.
    - numeric_vars (list): A list of numeric variables for which to calculate correlations.
    - target_var (str): The target variable for which correlations will be highlighted (default is 'Capped_incurred').
    - verbose (bool): If True, prints the correlations with the target variable in descending order.
    - figname (str or None): Full path to save the figure. If None, the plot will be displayed.

    Returns:
    - corrMat (pd.DataFrame): The correlation matrix of the numeric variables.
    """
    corrMat = df[numeric_vars].corr().abs()

    if verbose and target_var in corrMat.columns:
        print(f"Correlation with '{target_var}':")
        print(corrMat[target_var].sort_values(ascending=False))

    mask = np.triu(np.ones_like(corrMat, dtype=bool))
    plt.figure(figsize=(15, 10))
    sns.heatmap(corrMat, mask=mask, annot=True, cmap="coolwarm", 
                cbar_kws={"shrink": 0.7}, linewidths=0.4, fmt=".2f", 
                annot_kws={"size": 7})

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if figname:
        plt.savefig(figname, format='png')
    else:
        plt.show()

    return corrMat

####################################################################################################################

def plot_numeric_scatterplots(df, target, num_cols=3, verbose=False, figname=None):
    """
    Plot scatter plots for all numeric features against the target variable.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numeric features and the target variable.
    - target (str): The name of the target variable for the scatter plots.
    - num_cols (int): The number of columns for the subplot grid. Default is 3.
    - verbose (bool): If True, prints the list of numeric features.
    - figname (str or None): Full path to save the figure. If None, the plot will be displayed.

    Returns:
    - None. The function displays or saves scatter plots of numeric features against the target variable.
    """
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.drop(target)

    if verbose:
        print("Numeric Features:", numeric_features.tolist())

    num_features = len(numeric_features)
    num_rows = -(-num_features // num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(numeric_features):
        sns.scatterplot(x=df[feature], y=df[target], color='green', alpha=0.6, ax=axes[i])
        axes[i].set_title(f"'{feature}' vs '{target}'")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if figname:
        plt.savefig(figname, format='png')
    else:
        plt.show()

####################################################################################################################

def plot_categorical_columns_vs_target(df, target, categorical_columns, num_cols=3, figname=None):
    """
    Plots boxplots of the target variable against each categorical column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - target (str): The name of the continuous target variable.
    - categorical_columns (list): List of categorical column names.
    - num_cols (int): Number of subplot columns per row. Default is 3.
    - figname (str or None): Full path to save the figure. If None, the plot will be displayed.

    Returns:
    - None. The function displays or saves boxplots.
    """
    num_features = len(categorical_columns)
    num_rows = -(-num_features // num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(categorical_columns):
        sns.boxplot(x=df[column], y=df[target], palette='Set2', ax=axes[i])
        axes[i].set_title(f"'{target}' by '{column}'")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel(target)

        if df[column].nunique() > 5:
            axes[i].tick_params(axis='x', rotation=45)

        medians = df.groupby(column)[target].median()
        for tick, median in zip(axes[i].get_xticks(), medians):
            axes[i].text(tick, median, f"{median:.2f}", horizontalalignment='center', color='black', weight='semibold')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if figname:
        plt.savefig(figname, format='png')
    else:
        plt.show()

