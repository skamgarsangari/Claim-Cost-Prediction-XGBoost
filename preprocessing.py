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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import pickle
from EDA import plot_categorical_countplots, plot_numerical_histograms, plot_correlation_heatmap, plot_target_distribution

####################################################################################################################

def data_io(working_path, data_name, sheet_name = 'Data', verbose = False):
    """
    Load and prepare the data from an Excel file, ensuring necessary directories are created, 
    and checks are performed to avoid errors during file access.

    Parameters:
    - working_path (str): The main working directory path.
    - data_name (str): The name of the Excel file to be loaded.
    - sheet_name (str): The name of the sheet in the Excel file. Default is 'Data'.
    - verbose (bool): If True, prints dataset shape, column info, and summary statistics. Default is False.
    
    Returns:
    - df_cleaned (pd.DataFrame): The cleaned and loaded DataFrame from the Excel file.
    """

    # Check if the working directory exists to prevent any errors while accessing files.
    assert Path(working_path).is_dir(), f"No {working_path} directory exists!"

    # Define paths for data, model, and figures directories, ensuring they are created if they don't exist.
    data_path = Path(working_path) / "data"
    model_path = Path(working_path) / "model"
    fig_path = Path(working_path) / "figures"
    
    Path(data_path).mkdir(parents=True, exist_ok=True)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(fig_path).mkdir(parents=True, exist_ok=True)

    # Combine the working directory and file name to get the full path of the data file.
    data_file_path = data_path / data_name
    assert data_file_path.is_file(), f"No data file: {data_file_path}"  # Ensure the data file exists before proceeding.

    # Read the data from the Excel sheet.
    df_cleaned = pd.read_excel(data_file_path, sheet_name=sheet_name)  # Replace 'Data' if the sheet name differs.

    # Display basic information about the data if verbose is set to True.
    if verbose:
        print(df_cleaned.shape)  # Print the shape of the dataset (rows, columns)
        print(df_cleaned.info())  # Print data types and non-null counts for each column
        print(df_cleaned.describe())  # Print summary statistics for numerical columns

    return df_cleaned

####################################################################################################################

def apply_transformation(y_train, y_test, trans_method='none', plot=True):
    """
    Plots the distribution of the target variable (y_train and y_test) before and after applying a transformation.

    Parameters:
    - y_train (pd.Series or np.ndarray): The training target values.
    - y_test (pd.Series or np.ndarray): The test target values.
    - trans_method (str): The transformation method to apply to the target ('log', 'boxcox', or 'none').
    
    Returns:
    - y_train_transformed, y_test_transformed (np.ndarray): The transformed target values for training and testing sets.
    """

    if plot:
        # Plot the original distribution of y_train and y_test
        plt.figure(figsize=(10, 6))
        plt.hist(y_train, bins=30, alpha=0.5, label='y_train', color='b')  # Training target distribution
        plt.hist(y_test, bins=30, alpha=0.5, label='y_test', color='r')  # Test target distribution
        plt.title('Distribution of y_train and y_test (Original)')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    # Step 1: Apply log transformation or Box-Cox transformation
    if trans_method == 'log':
        # Apply log1p transformation (log(1 + x)) to handle zero values and skewness
        y_train_transformed = np.log1p(y_train)
        y_test_transformed = np.log1p(y_test)
    
    elif trans_method == 'boxcox':
        # Shift the values to ensure all values are positive before Box-Cox transformation
        shift_value = abs(min(min(y_train), min(y_test))) + 1  # Shift value to make all values positive
        y_train_shifted = y_train + shift_value  # Shifted training set
        y_test_shifted = y_test + shift_value  # Shifted test set

        # Apply Box-Cox transformation
        y_train_transformed, fitted_lambda = boxcox(y_train_shifted)  # Box-Cox for y_train
        y_test_transformed = boxcox(y_test_shifted, lmbda=fitted_lambda)  # Box-Cox for y_test
    
    else:
        # If no transformation is applied, keep the original target values
        y_train_transformed = y_train
        y_test_transformed = y_test

    if plot:
        # Plot the transformed distribution of y_train and y_test
        plt.figure(figsize=(10, 6))
        plt.hist(y_train_transformed, bins=30, alpha=0.5, label='y_train', color='b')  # Transformed training target
        plt.hist(y_test_transformed, bins=30, alpha=0.5, label='y_test', color='r')  # Transformed test target
        plt.title(f'Distribution of y_train and y_test (Transformed: {trans_method})')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    return y_train_transformed, y_test_transformed

####################################################################################################################







def preprocess_dataset(df, verbose=False):
    """
    Preprocess the dataset by renaming columns, creating new features, and applying transformations in place.

    Parameters:
    - df (pd.DataFrame): The original dataset to preprocess. The changes are applied directly to this dataset.
    - verbose (bool): If True, prints detailed information about the dataset after preprocessing.

    Returns:
    - None. The function modifies the dataset in place.
    """

    # Rename dataset columns by replacing spaces with underscores and capitalizing the first letter of each column name
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.capitalize()

    # Drop columns with only one unique value
    df.drop(columns=df.columns[df.nunique() == 1], inplace=True)

    # Print data info if verbose is set to True
    if verbose:
        print(df.info())

    # Convert 'Date_of_loss' to datetime format for time-based analysis
    df['Date_of_loss'] = pd.to_datetime(df['Date_of_loss'], errors='coerce')

    # Create a new feature 'Days_since_loss' to represent the time (in days) since the loss occurred
    current_date = pd.Timestamp.today()
    df['Days_since_loss'] = (current_date - df['Date_of_loss']).dt.days

    # Drop the 'Date_of_loss' column after transformation
    df.drop(columns=['Date_of_loss'], inplace=True)

    # Initialize a MinMaxScaler to scale 'Days_since_loss' between 0 and 1
    scaler = MinMaxScaler()
    df['Weight_days_since_loss'] = scaler.fit_transform(df[['Days_since_loss']])

    # Drop the original 'Days_since_loss' column after scaling
    df.drop(columns=['Days_since_loss'], inplace=True)

    # Apply sine and cosine transformations to 'Time_hour' for its cyclic nature
    df['Sin_time_hour'] = (np.sin(2 * np.pi * df['Time_hour'] / 24) + 1) / 2
    df['Cos_time_hour'] = (np.cos(2 * np.pi * df['Time_hour'] / 24) + 1) / 2

    # Drop the original 'Time_hour' column after transformation
    df.drop(columns=['Time_hour'], inplace=True)

    # If verbose is set to True, print detailed dataset info after preprocessing
    if verbose:
        print("Preprocessing completed. Here is the updated dataset:")
        print(df.info())


####################################################################################################################

def clean_and_impute_missing_values(df, verbose=False):
    """
    Clean and fill missing values in specific columns by replacing 'N/K' or variations with NaN, and filling NaN values 
    with the mode (most common value) in selected categorical columns. Modifies the DataFrame in place.

    Parameters:
    - df (pd.DataFrame): The DataFrame to clean and process. The changes are applied directly to this dataset.
    - verbose (bool): If True, prints unique values of specified columns for verification.

    Returns:
    - None. The function modifies the DataFrame in place.
    """

    # List of columns where 'N/K' should be replaced with NaN
    columns_to_replace = ['Weather_conditions', 'Location_of_incident', 'Vehicle_mobile']

    # Replace 'N/K' (and case variations) with np.nan in specific columns
    df[columns_to_replace] = df[columns_to_replace].replace(r'[nN]/[kK]', np.nan, regex=True)

    # Replace 'N/K' in 'Ph_considered_tp_at_fault' with a new category 'Not-Known'
    df['Ph_considered_tp_at_fault'] = df['Ph_considered_tp_at_fault'].replace(r'[nN]/[kK]', 'Not-Known', regex=True)

    # Optionally print unique values in 'Ph_considered_tp_at_fault' to verify replacements
    if verbose:
        print("Unique values in 'Ph_considered_tp_at_fault':", df['Ph_considered_tp_at_fault'].unique())

    # Replace any '#' values in 'Ph_considered_tp_at_fault' with NaN
    df['Ph_considered_tp_at_fault'] = df['Ph_considered_tp_at_fault'].replace('#', np.nan, regex=True)

    # List of columns where missing values will be filled with the mode
    missing_cat = ['Weather_conditions', 'Location_of_incident', 'Vehicle_mobile', 'Ph_considered_tp_at_fault']

    # Fill missing values in these categorical columns with the most common value (mode)
    df[missing_cat] = df[missing_cat].fillna(df[missing_cat].mode().iloc[0])

    # Optionally print a message to confirm completion if verbose is True
    if verbose:
        print("Missing values in the following columns were filled with the mode:", missing_cat)

####################################################################################################################

def drop_columns_and_update_numeric_features(df, columns_to_drop, verbose=False):
    """
    Drop specified columns from the DataFrame and update the numeric variables in place.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which columns will be dropped. Modifies the DataFrame in place.
    - columns_to_drop (list): A list of columns to be removed from the DataFrame.
    - verbose (bool): If True, prints the updated list of numeric variables for verification.

    Returns:
    - None. The function modifies the DataFrame and the numeric variables in place.
    """

    # Drop the specified columns from the DataFrame if they exist
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Select only the numerical columns (float64 and int64 data types)
    numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns

    # Update the list of numeric variables by excluding the dropped columns
    numeric_vars_update = [col for col in numeric_vars if col not in columns_to_drop]

    # If verbose is set to True, print the updated list of numeric variables
    if verbose:
        print("Updated list of numeric variables:", numeric_vars_update)


####################################################################################################################


def process_and_clean_numeric_features(df, numeric_vars, verbose=False, plot=True):
    """
    Cleans numerical data by filling missing values, removing unreasonable negative values, handling outliers, and 
    updating the dataset in place.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing numerical data to clean. The changes are applied directly to this dataset.
    - numeric_vars (list): A list of numerical columns to be processed.
    - verbose (bool): If True, prints missing values count, negative value information, and columns cleaned for verification.
    - plot (bool): If True, plots the distribution of numerical variables after cleaning.

    Returns:
    - None. The function modifies the DataFrame in place.
    """

    # Fill missing values in numerical columns with their median value
    df[numeric_vars] = df[numeric_vars].fillna(df[numeric_vars].median())

    # If verbose is set to True, print the count of missing values in each column
    if verbose:
        print("Missing values count after filling:", df.isnull().sum())






    # # Identify numeric columns that contain negative values
    # negative_columns = df[numeric_vars].columns[(df[numeric_vars] < 0).any()].tolist()

    # # Count how many negative values exist in each identified column
    # negative_counts = (df[negative_columns] < 0).sum()

    # # Define a threshold to decide which negative values should be removed
    # threshold = 0.00 * len(df)

    # # Identify columns where the number of negative values is within the threshold
    # columns_to_clean = negative_counts[negative_counts <= threshold].index.tolist()

    # # Remove rows with negative values in the identified columns
    # for col in columns_to_clean:
    #     df.drop(df[df[col] < 0].index, inplace=True)


    # Identify numeric columns that contain negative values
    negative_columns = df[numeric_vars].columns[(df[numeric_vars] < 0).any()].tolist()

    # Count how many negative values exist in each identified column
    negative_counts = (df[negative_columns] < 0).sum()

    if verbose:
        # Report the columns with negative values and their counts
        print(f"Found negative values in the following columns:")
        for col in negative_columns:
            print(f"- {col}: {negative_counts[col]} negative values")
    
    # Remove rows with negative values in the identified columns
    if negative_columns:
        df.drop(df[(df[negative_columns] < 0).any(axis=1)].index, inplace=True)
        if verbose:
            print(f"Removed rows with negative values from columns: {negative_columns}")
            print(f"Remaining rows after cleaning: {len(df)}")


    # Reset the index after removing rows to avoid gaps in the index
    df.reset_index(drop=True, inplace=True)

    # Drop columns with only one unique value
    unique_counts = df.nunique()
    columns_to_drop = unique_counts[unique_counts <= 1].index
    df.drop(columns=columns_to_drop, inplace=True)

    # Handle outliers by capping values above the 99th percentile for 'Notification_period'
    if 'Notification_period' in df.columns:
        upper_threshold = df['Notification_period'].quantile(0.99)
        df['Notification_period'] = np.where(df['Notification_period'] > upper_threshold, upper_threshold, df['Notification_period'])

    if plot:
        # Visualize the distribution of numerical variables after cleaning
        df[numeric_vars].hist(bins=50, figsize=(16, 12), log=True)
        plt.tight_layout()
        plt.show()



####################################################################################################################

def cap_incurred_and_drop_columns(df, threshold_target_value, targets_to_drop=None, verbose=False):
    """
    Caps the 'Incurred' column in the DataFrame based on the given threshold and drops specified target columns.
    Modifies the DataFrame in place.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'Incurred' column. The changes are applied directly to this dataset.
    - threshold_target_value (float): The quantile value to use for capping the 'Incurred' column (e.g., 0.9 for 90th percentile).
    - targets_to_drop (list, optional): List of target columns to drop (e.g., 'Incurred', 'Capped_incurred'). If None, default targets will be used.
    - verbose (bool): If True, prints the upper threshold used for capping and the columns dropped.

    Returns:
    - None. The function modifies the DataFrame in place.
    """

    # Step 1: Calculate the upper threshold for the 'Incurred' column based on the given threshold value
    upper_threshold_target = df['Incurred'].quantile(threshold_target_value)

    # Step 2: Create a dynamic column name reflecting the threshold value (e.g., "Incurred_capped_90p")
    target_fname = f'Incurred_capped_{threshold_target_value * 100}p'

    # Step 3: Cap the 'Incurred' values using the clip() method
    df[target_fname] = df['Incurred'].clip(upper=upper_threshold_target)

    # Step 4: Drop target columns ('Incurred', 'Capped_incurred') from the DataFrame if they exist
    if targets_to_drop is None:
        targets_to_drop = ['Incurred', 'Capped_incurred']  # Default columns to drop
    
    df.drop(columns=[col for col in targets_to_drop if col in df.columns], inplace=True)

    # Optional verbose mode for detailed output
    if verbose:
        print(f"Capped 'Incurred' values at the {threshold_target_value*100}th percentile: {upper_threshold_target}")
        print(f"Columns dropped: {targets_to_drop}")


####################################################################################################################

def prepare_data_for_modeling(df, target, weights, encoding_methods, decay_func=True, split_test_size=0.35, verbose = False):
    """
    Prepares data for modeling by splitting into training and test sets, applying sample weights, and encoding categorical variables.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - target (str): The name of the target variable.
    - weights (str): The name of the column used for sample weighting.
    - encoding_methods (dict): Dictionary specifying encoding methods for categorical columns (e.g., 'onehot', 'label').
    - split_test_size (float): The proportion of the dataset to include in the test split. Default is 0.35.
    - decay_func (bool): If True, applies exponential decay to sample weights. Default is True.
    - verbose (bool): If True, prints missing values and sample weights.

    Returns:
    - X_train, X_test, y_train, y_test, train_weights, test_weights (pd.DataFrames): Training and test sets with features, target, and weights.
    """

    # Create a copy of the DataFrame for modeling
    df_model = df.copy()

    # Define X (features) and y (target)
    X = df_model.drop(columns=[target, weights])  # Drop target and weight columns from feature set
    y = df_model[target]  # Define the target variable

    # Create sample weights by reversing the 'Days_since_loss' weights

    if decay_func:
        # Set an optional decay rate, default is 1 (no decay)
        decay_rate = 0.5  # You can adjust this value (e.g., 0.5, 0.1, etc.) to increase the decay effect
        # Exponentially decay weights so that recent claims are weighted more
        sample_weights = np.exp(-decay_rate * df_model[weights])
    else:
        # Create sample weights by differentiating the 'Days_since_loss' weights from 1 (recent claims are weighted more)
        sample_weights = 1 - df_model[weights]


    # Display the first 10 sample weights if verbose
    if verbose:
        print(sample_weights.head(10))

    # If verbose is enabled, print any missing values in the feature set
    if verbose:
        print("Missing values in features:\n", X.isnull().sum())

    # Split the data into training and testing sets (including features, target, and weights)
    X_train, X_test, y_train, y_test, train_weights, test_weights = train_test_split(
        X, y, sample_weights, test_size=split_test_size, random_state=42
    )

    # Initialize a dictionary to store encoders for each categorical feature
    encoders = {}

    # Loop through the encoding methods and apply them to both training and test sets
    for col, encoding in encoding_methods.items():
        
        if encoding == 'label':
            le = LabelEncoder()
            
            # Apply label encoding and store the encoder for future reference
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            encoders[col] = le

        elif encoding == 'onehot':
            ohe = OneHotEncoder(sparse_output=False, drop='first')  # Drop the first level to avoid dummy variable trap
            
            # Apply one-hot encoding to both training and testing sets
            X_train_ohe = ohe.fit_transform(X_train[[col]])
            X_test_ohe = ohe.transform(X_test[[col]])
            
            # Convert the one-hot encoded arrays back to DataFrames and merge them with the original sets
            X_train_ohe = pd.DataFrame(X_train_ohe, index=X_train.index, columns=ohe.get_feature_names_out([col]))
            X_test_ohe = pd.DataFrame(X_test_ohe, index=X_test.index, columns=ohe.get_feature_names_out([col]))
            
            # Drop the original column and add the new one-hot encoded columns
            X_train = X_train.drop(columns=[col]).join(X_train_ohe)
            X_test = X_test.drop(columns=[col]).join(X_test_ohe)
            
            encoders[col] = ohe

        else:
            raise ValueError(f"Unknown encoding method: {encoding}")

    # Return the prepared training and testing sets along with the weights
    return X_train, X_test, y_train, y_test, train_weights, test_weights

####################################################################################################################

def train_xgboost_model(X_train, y_train, train_weights, X_test, param_dist, n_iter=100, scoring='neg_mean_absolute_error', cv=5, verbose=2, random_state=42):
    """
    Trains an XGBoost model with hyperparameter tuning using RandomizedSearchCV, and makes predictions on the test set.

    Parameters:
    - X_train (pd.DataFrame): The training feature set.
    - y_train (pd.Series or np.ndarray): The transformed target variable for training.
    - train_weights (pd.Series or np.ndarray): Sample weights for training.
    - X_test (pd.DataFrame): The test feature set.
    - param_dist (dict): The hyperparameter grid to search for the XGBoost model.
    - n_iter (int): The number of iterations for RandomizedSearchCV (default is 100).
    - scoring (str): The scoring metric for RandomizedSearchCV (default is 'neg_mean_absolute_error').
    - cv (int): The number of cross-validation folds (default is 5).
    - verbose (int): The verbosity level of the search (default is 2).
    - random_state (int): The random state for reproducibility (default is 42).

    Returns:
    - best_xgboost_model (xgb.XGBRegressor): The best XGBoost model found by RandomizedSearchCV.
    - y_pred (np.ndarray): Predictions made by the best model on the test set.
    """

    # Define the XGBoost model
    xgboost_model = xgb.XGBRegressor(random_state=random_state)

    # Perform hyperparameter tuning with RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgboost_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=-1
    )

    # Fit the model on the training data with sample weights
    random_search.fit(X_train, y_train, sample_weight=train_weights)

    # Output the best parameters and best score from RandomizedSearchCV
    print("[DEBUG] - Best Parameters:", random_search.best_params_)
    print("[DEBUG] - Best CV Score (neg MAE):", random_search.best_score_)

    # Extract the best model from RandomizedSearchCV
    best_xgboost_model = random_search.best_estimator_

    # Make predictions on the test set using the best model
    y_pred = best_xgboost_model.predict(X_test)

    return best_xgboost_model, y_pred

####################################################################################################################

def evaluate_model_and_save(
    y_test, y_pred_transformed, best_model, X_train, trans_method='none', 
    fitted_lambda=None, shift_value=0, model_save_path='best_model.pkl', plot=True):
    """
    Evaluates the model's performance by calculating R², MAE, and RMSE, and optionally reverses transformations on the predictions.
    Additionally, it calculates and optionally plots residuals, and saves the model and transformation parameters in a pickle file.

    Parameters:
    - y_test (pd.Series or np.ndarray): The original target values from the test set.
    - y_pred_transformed (np.ndarray): The predicted values after transformation.
    - best_model: The best model (e.g., XGBoost) used for feature importance extraction.
    - X_train (pd.DataFrame): The training feature set used to get the feature names.
    - trans_method (str): The transformation method used ('log', 'boxcox', or 'none').
    - fitted_lambda (float): The lambda parameter for Box-Cox transformation (required if `boxcox` is used).
    - shift_value (float): The shift value used for the Box-Cox transformation (required if `boxcox` is used).
    - model_save_path (str): The file path where the best model and associated parameters will be saved.
    - plot (bool): If True, plots feature importance and residuals.

    Returns:
    - metrics (dict): A dictionary containing R², MAE, MSE, and RMSE values.
    - feature_importance_df (pd.DataFrame): A DataFrame of feature importances.
    """
    # Reverse the transformation on the predictions (if necessary)
    if trans_method == 'log':
        y_pred = np.expm1(y_pred_transformed)  # Inverse log transformation
    elif trans_method == 'boxcox':
        y_pred_shifted = inv_boxcox(y_pred_transformed, fitted_lambda)  # Inverse Box-Cox transformation
        y_pred = y_pred_shifted - shift_value
    else:
        y_pred = y_pred_transformed

    try:
        # Calculate R² score
        r2 = r2_score(y_test, y_pred)
        print(f"R² Score after {trans_method} transformation: {r2}")
    except:
        r2 = None
        print("R² Score could not be calculated.")


    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Print evaluation metrics
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Get feature importances from the model
    importances = best_model.feature_importances_

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Print the most important features
    print(feature_importance_df)

    if plot:
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title('Feature Importance (XGBoost)')
        plt.tight_layout()
        plt.show()

        # Calculate residuals
        residuals = y_test - y_pred

        # Plot residuals scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
        plt.axhline(0, color='r', linestyle='--')  # Add a horizontal line at 0 for reference
        plt.title('Residuals Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.show()

        # Plot the distribution of residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='blue')
        plt.title('Distribution of Residuals')
        plt.xlabel('Residuals')
        plt.grid(True)
        plt.show()

    # Save the model and parameters
    metrics = {
        'R²': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }

    # Ensure the model subdirectory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save the best model and transformation parameters to a pickle file
    model_data = {
        'best_model': best_model,
        'feature_names': X_train.columns.tolist(),
        'trans_method': trans_method,
        'fitted_lambda': fitted_lambda,
        'shift_value': shift_value
    }

    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model and parameters saved to {model_save_path}")

    return metrics, feature_importance_df


####################################################################################################################


def evaluate_max_depth(X_train, y_train, X_test, y_test, list_depths=np.arange(1, 51), plot=True):
    """
    Evaluates the performance of an XGBoost model over a range of max_depth values by calculating MAE for both training and validation sets.
    Optionally, plots the results of MAE vs. max_depth.

    Parameters:
    - X_train (pd.DataFrame): The training feature set.
    - y_train (pd.Series or np.ndarray): The target values for the training set.
    - X_test (pd.DataFrame): The test feature set.
    - y_test (pd.Series or np.ndarray): The target values for the test set.
    - list_depths (np.ndarray): The range of max_depth values to evaluate (default is 1 to 50).
    - plot (bool): If True, plots MAE vs. max_depth. Default is True.

    Returns:
    - lis_cv_score_train, lis_cv_score_test (list): Lists of MAE scores for training and validation sets.
    """

    # Lists to store MAE scores for training and validation sets
    lis_cv_score_train = []
    lis_cv_score_test = []

    # Loop through different max_depth values to evaluate model performance
    for depth in tqdm(list_depths):  # tqdm used for progress tracking
        # Define the parameters for XGBoost
        params = {
            'colsample_bytree': 0.5,
            'subsample': 0.5,
            'learning_rate': 0.01,
            'max_depth': depth,  # Max depth changes in each iteration
            'min_child_weight': 1,
            'objective': 'reg:squarederror',  # Regression objective
            'seed': 2016  # Random seed for reproducibility
        }

        # Initialize the XGBRegressor with the current set of parameters
        xgb_model = xgb.XGBRegressor(**params, n_estimators=700, random_state=2016)
        
        # Fit the model on the training set
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)  # Train without verbose output
        
        # Evaluate on the training set
        y_train_pred = xgb_model.predict(X_train)
        cv_score_train = mean_absolute_error(y_train, y_train_pred)  # Calculate MAE for training set
        lis_cv_score_train.append(cv_score_train)  # Append MAE score
        
        # Evaluate on the validation (test) set
        y_test_pred = xgb_model.predict(X_test)
        cv_score_test = mean_absolute_error(y_test, y_test_pred)  # Calculate MAE for validation set
        lis_cv_score_test.append(cv_score_test)  # Append MAE score

    if plot:
        # Plot the results of training and validation MAE scores
        plt.figure(figsize=(10, 6))
        plt.plot(list_depths, lis_cv_score_train, label="Train", marker='o', color='blue')  # Training set MAE
        plt.plot(list_depths, lis_cv_score_test, label="Validation", marker='o', color='red')  # Validation set MAE
        plt.xlabel("Max Depth")  # x-axis label
        plt.ylabel("MAE Score")  # y-axis label
        plt.title("MAE vs Max Depth for Train and Validation Sets")  # Plot title
        plt.legend()  # Show legend
        plt.grid(True)  # Enable grid for better readability
        plt.tight_layout()  # Adjust layout
        plt.show()  # Display the plot

    return lis_cv_score_train, lis_cv_score_test


####################################################################################################################

def preprocess_new_data(new_data, encoding_methods, columns_to_drop=None, mode='train', verbose=False, plot=False, target=None):
    """
    Applies all the necessary preprocessing steps to the new test data before deploying the model or during training.
    The function mirrors the preprocessing performed on the training data, including cleaning, handling missing values, 
    dropping unnecessary columns, and encoding categorical variables. Depending on the mode ('train' or 'deploy'), 
    it drops specific columns (e.g., 'incurred' and weight columns). If in 'train' mode, it ensures that columns 
    containing 'incurred' are dropped, except for the target column.

    Parameters:
    - new_data (pd.DataFrame): The new test data to preprocess.
    - encoding_methods (dict): Dictionary specifying encoding methods for categorical columns (e.g., 'onehot', 'label').
    - columns_to_drop (list, optional): List of columns to drop from the dataset, including unnecessary features or target variables.
    - mode (str): Specifies whether preprocessing is for 'train' or 'deploy'. If 'deploy', drops 'incurred' and weight columns.
    - verbose (bool): If True, prints detailed information during preprocessing.
    - plot (bool): If True, generates and displays distribution plots for numeric data.
    - target (str): The target variable to retain when dropping columns containing 'incurred' during training.

    Returns:
    - new_data (pd.DataFrame): The preprocessed data ready for the model.
    """

    # Step 1: General preprocessing (basic cleaning, feature creation, transformations)
    preprocess_dataset(new_data, verbose=verbose)

    # Step 2: Clean and fill missing values
    clean_and_impute_missing_values(new_data, verbose=verbose)

    # Step 3: Drop unnecessary columns (e.g., target variables, irrelevant features)
    if columns_to_drop is not None:
        drop_columns_and_update_numeric_features(new_data, columns_to_drop, verbose=verbose)

    # Step 4: Depending on the mode ('train' or 'deploy'), decide whether to drop columns containing 'incurred'
    if 'deploy' in mode.lower():  # Drop 'incurred' and weight columns in deployment mode
        # Drop all columns containing 'incurred' (case-insensitive)
        incurred_columns = [col for col in new_data.columns if 'incurred' in col.lower()]
        if verbose:
            print(f"Deployment mode: Dropping columns containing 'incurred': {incurred_columns}")
        new_data.drop(columns=incurred_columns, inplace=True)

        # Also drop the 'weight' column used during training
        if 'Weight_days_since_loss' in new_data.columns:
            if verbose:
                print(f"Deployment mode: Dropping column 'weight_days_since_loss'")
            new_data.drop(columns=['Weight_days_since_loss'], inplace=True)
    elif 'train' in mode.lower():  # Training mode doesn't drop the target column
        # Drop all columns containing 'incurred', except the target variable
        incurred_columns = [col for col in new_data.columns if 'incurred' in col.lower() and col != target]
        if verbose:
            print(f"Training mode: Dropping columns containing 'incurred' (except target '{target}'): {incurred_columns}")
        new_data.drop(columns=incurred_columns, inplace=True)

    # Step 5: Clean numerical columns, handle missing values, and outliers
    numeric_vars = new_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    process_and_clean_numeric_features(new_data, numeric_vars, verbose=verbose, plot=plot)

    # Step 6: Apply encoding to categorical columns
    encoders = {}

    for col, encoding in encoding_methods.items():
        if encoding == 'label':
            # Apply label encoding to categorical columns
            le = LabelEncoder()
            new_data[col] = le.fit_transform(new_data[col].astype(str))
            encoders[col] = le
        elif encoding == 'onehot':
            # Apply one-hot encoding to categorical columns
            ohe = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid dummy variable trap
            new_data_ohe = ohe.fit_transform(new_data[[col]])
            new_data_ohe = pd.DataFrame(new_data_ohe, index=new_data.index, columns=ohe.get_feature_names_out([col]))
            new_data = new_data.drop(columns=[col]).join(new_data_ohe)
            encoders[col] = ohe
        else:
            # Raise an error if an unknown encoding method is provided
            raise ValueError(f"Unknown encoding method: {encoding}")

    return new_data

