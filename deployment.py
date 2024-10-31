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
from preprocessing import data_io, apply_transformation, preprocess_dataset, clean_and_impute_missing_values, drop_columns_and_update_numeric_features, process_and_clean_numeric_features, cap_incurred_and_drop_columns,preprocess_new_data
from EDA import plot_categorical_countplots, plot_numerical_histograms, plot_correlation_heatmap, plot_target_distribution


####################################################################################################################

def deploy_model(excel_path, training_param_path = 'training_params.pkl', model_path='best_model.pkl'):
    """
    Deploys a trained model on new test data provided in an Excel file. It applies preprocessing steps,
    handles any necessary column transformations, and generates predictions based on the input data.

    Parameters:
    - excel_path (str): The path to the Excel file containing the new test data.
                       The Excel file must have a sheet called 'Data' or the appropriate sheet name.
    - model_path (str): The path to the saved model pickle file. Defaults to 'best_model.pkl'.
    - training_param_path (str): The path to the saved training parameters pickle file. Defaults to 'training_params.pkl'.

    Returns:
    - predictions (pd.Series): The predictions made by the model on the new test data, after all necessary transformations.
    """

    # Load the new Excel file as a DataFrame
    new_data = pd.read_excel(excel_path, sheet_name='Data')  # Assuming 'Data' is the correct sheet name


    # Load the saved model and associated parameters from the pickle file
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    best_model = model_data['best_model']  # 
    feature_names = model_data['feature_names']  # The features used in training
    trans_method = model_data['trans_method']  # Transformation applied to the target variable during training
    fitted_lambda = model_data.get('fitted_lambda', None)  # Lambda for Box-Cox, if used
    shift_value = model_data.get('shift_value', 0)  # Shift value for Box-Cox, if used


    # Load the saved model and associated parameters from the pickle file
    with open(training_param_path, 'rb') as f:
        training_data = pickle.load(f)
    encoding_methods = training_data['encoding_methods']  # Encoding methods for categorical columns
    columns_to_drop = training_data['columns_to_drop']  # Columns dropped during training
    target = training_data['target']  # The target column
    split_test_size = training_data['split_test_size']  # Test set split size
    threshold_target_value = training_data['threshold_target_value']  # Threshold value for capping


    # Print a summary of the parameters read from the pickle file
    print("\n--- Model Training Parameters Summary ---")
    print(f"Encoding Methods: {encoding_methods}")
    print(f"transformation method: {trans_method}")
    print(f"Columns Dropped: {columns_to_drop}")
    print(f"Transformation Method: {trans_method}")
    print(f"Target Column: {target}")
    print(f"Test Set Split Size: {split_test_size}")
    print(f"Threshold Target Value: {threshold_target_value}")
    print("-----------------------------------------\n")


    # Ensure encoding methods for categorical variables are provided
    assert encoding_methods is not None, "Please provide encoding methods for categorical columns."

    # Step 1: Preprocess the new data (apply the same steps used in training)
    new_data = preprocess_new_data(new_data, encoding_methods, columns_to_drop=columns_to_drop, mode='deploy')

    # Step 2: Apply the model to the preprocessed new data to generate predictions
    predictions_transformed = best_model.predict(new_data)

    # Step 3: Reverse any transformation applied to the target (e.g., log or Box-Cox transformation)
    if trans_method == 'log':
        predictions = np.expm1(predictions_transformed)  # Inverse of log(1+x) transformation
    elif trans_method == 'boxcox':
        predictions_shifted = inv_boxcox(predictions_transformed, fitted_lambda)  # Inverse Box-Cox transformation
        predictions = predictions_shifted - shift_value  # Subtract the shift value used during training
    else:
        predictions = predictions_transformed  # If no transformation was applied, use raw predictions

    # Return the predictions as a pandas Series for easier manipulation
    return pd.Series(predictions, name='Predictions')


####################################################################################################################

if __name__ == "__main__":
    """
    Main execution flow for the project workflow. It handles:
    - Data preprocessing and EDA for the original dataset.
    - Training a machine learning model with optional hyperparameter tuning.
    - Deploying the trained model to new data for making predictions.
    """

    model_path='/Users/saeideh/project2/model/best_model.pkl'
    training_param_path='/Users/saeideh/project2/model/training_params.pkl'
    deploy_data_path='/Users/saeideh/project2/data/deployments_data.xlsx'

    # Deploy the model and make predictions on new test data
    predictions = deploy_model(
        excel_path=deploy_data_path, 
        model_path=model_path,
        training_param_path=training_param_path
    )
    print(f'prediction of claim based on the model: \n {predictions}')


