import os
import sys
import pandas as pd
import numpy as np
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
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(n_cols, int) or n_cols <= 0:
        raise ValueError("n_cols must be a positive integer")
    
    # Get categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_features) == 0:
        print("No categorical features found in the DataFrame.")
        return
    
    # Create subplot grid
    n_rows = -(-len(categorical_features) // n_cols)  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4), sharey=False)
    
    # Handle single row/column case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each categorical feature
    for i, col in enumerate(categorical_features):
        try:
            # Count plot with error handling for too many categories
            if df[col].nunique() > 30:
                print(f"Warning: Column '{col}' has {df[col].nunique()} unique values, showing top 30.")
                value_counts = df[col].value_counts().nlargest(30)
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
            else:
                sns.countplot(x=col, data=df, ax=axes[i])
            
            axes[i].set_title(f'Count plot of {col}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
        except Exception as e:
            print(f"Error plotting {col}: {str(e)}")
            axes[i].set_title(f'Error plotting {col}')

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save or display the figure
    if figname:
        try:
            plt.savefig(figname, format='png', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {figname}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
    else:
        plt.show()


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
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    
    # Select numeric variables
    numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_vars) == 0:
        print("No numeric features found in the DataFrame.")
        return
    
    # Determine layout based on number of variables
    n_cols = min(3, len(numeric_vars))
    n_rows = -(-len(numeric_vars) // n_cols)  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    
    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot histograms for each numeric variable
    for i, col in enumerate(numeric_vars):
        try:
            ax = axes[i] if i < len(axes) else None
            if ax is not None:
                # Handle very large values by showing in thousands/millions
                data = df[col]
                scale_factor = 1
                label_suffix = ""
                
                if data.max() > 1_000_000:
                    scale_factor = 1_000_000
                    label_suffix = " (in millions)"
                elif data.max() > 1_000:
                    scale_factor = 1_000
                    label_suffix = " (in thousands)"
                
                scaled_data = data / scale_factor
                
                # Plot histogram
                ax.hist(scaled_data.dropna(), bins=bins, log=log_scale)
                ax.set_title(f'Histogram of {col}')
                ax.set_xlabel(f'{col}{label_suffix}')
                ax.set_ylabel('Frequency' + (' (log scale)' if log_scale else ''))
                
                # Add mean and median lines
                mean_val = scaled_data.mean()
                median_val = scaled_data.median()
                ax.axvline(mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='g', linestyle='-.', alpha=0.5, label=f'Median: {median_val:.2f}')
                ax.legend(loc='best', fontsize='small')
        except Exception as e:
            print(f"Error plotting histogram for {col}: {str(e)}")
            if i < len(axes):
                axes[i].set_title(f'Error plotting {col}')
    
    # Remove empty subplots
    for j in range(len(numeric_vars), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save or display the figure
    if figname:
        try:
            plt.savefig(figname, format='png', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {figname}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
    else:
        plt.show()


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
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError(f"Target variable '{target}' must be numeric")
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Calculate basic statistics
    mean_val = df[target].mean()
    median_val = df[target].median()
    std_val = df[target].std()
    
    # Plot histogram with KDE
    try:
        sns.histplot(df[target].dropna(), color='g', kde=True, bins=bins)
        plt.title(f'Distribution of {target}')
        plt.xlabel(target)
        plt.ylabel('Frequency')
        
        # Add vertical lines for mean and median
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='b', linestyle='-.', label=f'Median: {median_val:.2f}')
        
        # Add text with statistics
        stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd Dev: {std_val:.2f}"
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    va='top', fontsize=10)
        
        plt.legend()
        plt.tight_layout()
        
        # Save or display the figure
        if figname:
            plt.savefig(figname, format='png', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {figname}")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting target distribution: {str(e)}")


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
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not all(var in df.columns for var in numeric_vars):
        missing_vars = [var for var in numeric_vars if var not in df.columns]
        raise ValueError(f"Variables not found in DataFrame: {missing_vars}")
    if not all(pd.api.types.is_numeric_dtype(df[var]) for var in numeric_vars):
        non_numeric = [var for var in numeric_vars if not pd.api.types.is_numeric_dtype(df[var])]
        raise TypeError(f"Non-numeric variables: {non_numeric}")
    if not isinstance(num_cols, int) or num_cols <= 0:
        raise ValueError("num_cols must be a positive integer")
    
    # Create subplot grid
    num_vars = len(numeric_vars)
    if num_vars == 0:
        print("No numeric variables to plot.")
        return
    
    num_rows = -(-num_vars // num_cols)  # Ceiling division
    
    # Handle single subplot case
    if num_rows == 1 and num_cols == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = np.array([ax])
    else:
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot boxplots for each numeric variable
    for i, var in enumerate(numeric_vars):
        try:
            ax = axes[i] if i < len(axes) else None
            if ax is not None:
                # Calculate statistics for annotation
                mean_val = df[var].mean()
                median_val = df[var].median()
                
                # Create boxplot
                sns.boxplot(y=df[var], ax=ax)
                ax.set_title(f'Boxplot of {var}')
                ax.set_xlabel('')
                ax.set_ylabel('Value')
                
                # Add annotations for mean and median
                ax.axhline(mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
                ax.axhline(median_val, color='g', linestyle='-.', alpha=0.5, label=f'Median: {median_val:.2f}')
                
                # Show legend only if there's room
                if df[var].nunique() < 20:
                    ax.legend(loc='best', fontsize='small')
        except Exception as e:
            print(f"Error plotting boxplot for {var}: {str(e)}")
            if i < len(axes):
                axes[i].set_title(f'Error plotting {var}')
    
    # Remove empty subplots
    for j in range(len(numeric_vars), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save or display the figure
    if figname:
        try:
            plt.savefig(figname, format='png', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {figname}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
    else:
        plt.show()


def plot_correlation_heatmap(df, numeric_vars, target_var='Incurred_capped_97.5p', verbose=False, figname=None):
    """
    Plot a correlation heatmap for numeric variables, improving readability for crowded plots.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numeric variables.
    - numeric_vars (list): A list of numeric variables for which to calculate correlations.
    - target_var (str): The target variable for which correlations will be highlighted.
    - verbose (bool): If True, prints the correlations with the target variable in descending order.
    - figname (str or None): Full path to save the figure. If None, the plot will be displayed.

    Returns:
    - corrMat (pd.DataFrame): The correlation matrix of the numeric variables.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    # Check that all variables are in the DataFrame
    missing_vars = [var for var in numeric_vars if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Variables not found in DataFrame: {missing_vars}")
    
    # Ensure all variables are numeric
    non_numeric = [var for var in numeric_vars if not pd.api.types.is_numeric_dtype(df[var])]
    if non_numeric:
        raise TypeError(f"Non-numeric variables found: {non_numeric}")
    
    # Check if target_var is in numeric_vars
    if target_var not in numeric_vars and target_var != 'Incurred_capped_97.5p':
        print(f"Warning: Target variable '{target_var}' not in numeric_vars list. Adding it.")
        if target_var in df.columns and pd.api.types.is_numeric_dtype(df[target_var]):
            numeric_vars = numeric_vars + [target_var]
        else:
            print(f"Warning: Target variable '{target_var}' not found or not numeric. Proceeding without it.")
            target_var = None
    
    try:
        # Calculate correlation matrix
        corrMat = df[numeric_vars].corr().abs()
        
        # Print correlations with target variable if requested
        if verbose and target_var in corrMat.columns:
            print(f"Correlation with '{target_var}':")
            target_corrs = corrMat[target_var].sort_values(ascending=False)
            # Format for better readability
            for var, corr in target_corrs.items():
                print(f"{var:<30} {corr:.4f}")
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corrMat, dtype=bool))
        
        # Create heatmap
        plt.figure(figsize=(15, 10))
        
        # Adjust annotation size and format based on number of variables
        annot_kws = {"size": 7}
        if len(numeric_vars) > 20:
            annot_kws["size"] = 5
        if len(numeric_vars) > 30:
            # Too many variables, show without annotations
            annot = False
            print("Warning: Too many variables for annotations. Displaying heatmap without values.")
        else:
            annot = True
        
        # Plot heatmap
        sns.heatmap(corrMat, mask=mask, annot=annot, cmap="coolwarm", 
                    cbar_kws={"shrink": 0.7}, linewidths=0.4, fmt=".2f", 
                    annot_kws=annot_kws)
        
        plt.title("Correlation Heatmap")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save or display the figure
        if figname:
            plt.savefig(figname, format='png', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {figname}")
        else:
            plt.show()
        
        return corrMat
    
    except Exception as e:
        print(f"Error calculating correlation matrix: {str(e)}")
        return None


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
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError(f"Target variable '{target}' must be numeric")
    if not isinstance(num_cols, int) or num_cols <= 0:
        raise ValueError("num_cols must be a positive integer")
    
    # Get numeric features excluding the target
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.drop(target, errors='ignore')
    
    if len(numeric_features) == 0:
        print("No numeric features found (excluding target).")
        return
    
    if verbose:
        print("Numeric Features:", numeric_features.tolist())
    
    # Create subplot grid
    num_features = len(numeric_features)
    num_rows = -(-num_features // num_cols)  # Ceiling division
    
    # Handle single subplot case
    if num_features == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot scatterplots for each numeric feature
    for i, feature in enumerate(numeric_features):
        try:
            ax = axes[i] if i < len(axes) else None
            if ax is not None:
                # Check if there are too many points (for performance)
                if len(df) > 5000:
                    # Sample for large datasets
                    sample_size = 5000
                    print(f"Warning: Large dataset detected. Sampling {sample_size} points for scatterplot of '{feature}'.")
                    sample_df = df.sample(sample_size, random_state=42)
                    sns.scatterplot(x=sample_df[feature], y=sample_df[target], color='green', alpha=0.6, ax=ax)
                    ax.set_title(f"'{feature}' vs '{target}' (sampled)")
                else:
                    sns.scatterplot(x=df[feature], y=df[target], color='green', alpha=0.6, ax=ax)
                    ax.set_title(f"'{feature}' vs '{target}'")
                
                # Add regression line
                sns.regplot(x=df[feature], y=df[target], scatter=False, color='red', ax=ax)
                
                # Add correlation annotation
                corr = df[[feature, target]].corr().iloc[0, 1]
                ax.annotate(f"Corr: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                            va='top', fontsize=10)
                
                ax.set_xlabel(feature)
                ax.set_ylabel(target)
        except Exception as e:
            print(f"Error plotting scatterplot for {feature}: {str(e)}")
            if i < len(axes):
                axes[i].set_title(f"Error plotting '{feature}' vs '{target}'")
    
    # Remove empty subplots
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save or display the figure
    if figname:
        try:
            plt.savefig(figname, format='png', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {figname}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
    else:
        plt.show()


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
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError(f"Target variable '{target}' must be numeric")
    
    # Check that all categorical columns are in the DataFrame
    missing_columns = [col for col in categorical_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    
    if len(categorical_columns) == 0:
        print("No categorical columns provided.")
        return
    
    # Create subplot grid
    num_rows = -(-len(categorical_columns) // num_cols)  # Ceiling division
    
    # Handle single subplot case
    if len(categorical_columns) == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot boxplots for each categorical column
    for i, column in enumerate(categorical_columns):
        try:
            ax = axes[i] if i < len(axes) else None
            if ax is not None:
                # Handle columns with too many categories
                if df[column].nunique() > 20:
                    print(f"Warning: Column '{column}' has {df[column].nunique()} unique values. Showing top 20 by frequency.")
                    # Get top categories by frequency
                    top_categories = df[column].value_counts().nlargest(20).index
                    # Filter dataframe to include only top categories
                    plot_df = df[df[column].isin(top_categories)]
                    ax.set_title(f"'{target}' by '{column}' (top 20 categories)")
                else:
                    plot_df = df
                    ax.set_title(f"'{target}' by '{column}'")
                
                # Create boxplot
                sns.boxplot(x=plot_df[column], y=plot_df[target], palette='Set2', ax=ax)
                
                # Handle missing values for median calculation
                medians = plot_df.groupby(column)[target].median()
                
                # Add median labels if there aren't too many categories
                if len(medians) <= 20:
                    for tick, label in enumerate(ax.get_xticklabels()):
                        category = label.get_text()
                        if category in medians.index:
                            median = medians[category]
                            ax.text(tick, median, f"{median:.2f}", 
                                   horizontalalignment='center', color='black', weight='semibold')
                
                ax.set_xlabel(column)
                ax.set_ylabel(target)
                
                # Rotate x-axis labels if needed
                if df[column].nunique() > 5:
                    ax.tick_params(axis='x', rotation=45)
                
                # Set reasonable y-axis limits to avoid extreme outliers dominating the plot
                q1, q3 = plot_df[target].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = max(plot_df[target].min(), q1 - 3 * iqr)
                upper_bound = min(plot_df[target].max(), q3 + 3 * iqr)
                margin = (upper_bound - lower_bound) * 0.1
                ax.set_ylim(lower_bound - margin, upper_bound + margin)
        
        except Exception as e:
            print(f"Error plotting boxplot for {column}: {str(e)}")
            if i < len(axes):
                axes[i].set_title(f"Error plotting '{target}' by '{column}'")
    
    # Remove empty subplots
    for j in range(len(categorical_columns), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save or display the figure
    if figname:
        try:
            plt.savefig(figname, format='png', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {figname}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
    else:
        plt.show()
