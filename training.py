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


def train_model(excel_path, training_param_path='training_params.pkl', model_save_path = 'best_model.pkl', encoding_methods=None, columns_to_drop=None, 
                weights='Weight_days_since_loss', param_dist=None, 
                n_iter=100, scoring='neg_mean_absolute_error', cv=5, verbose=2, random_state=42, 
                threshold_target_value=0.9, decay_func=True, split_test_size=0.35, trans_method='log'):
    """
    Preprocesses the dataset, generates the target variable based on the threshold value, verifies its correctness,
    trains the XGBoost model, and saves the trained model along with key parameters. It handles all necessary 
    column transformations, hyperparameter tuning, and performance evaluations.

    Parameters:
    - excel_path (str): Path to the Excel file containing the training data.
    - training_param_path (str): Path where the best trained model and parameters will be saved.
    - model_save_path (str): Path where the best trained model will be saved.
    - encoding_methods (dict): Dictionary specifying encoding methods for categorical columns (e.g., 'onehot', 'label').
    - columns_to_drop (list): List of columns to drop from the dataset (e.g., target variable columns or unnecessary features).
    - weights (str): The name of the column used for sample weighting.
    - param_dist (dict): Hyperparameter grid for XGBoost. If not provided, uses a default set of parameters.
    - n_iter (int): Number of iterations for hyperparameter tuning (default is 100).
    - scoring (str): Scoring metric for hyperparameter tuning (default is 'neg_mean_absolute_error').
    - cv (int): Number of cross-validation folds (default is 5).
    - verbose (int): Verbosity level for training output.
    - random_state (int): Random seed for reproducibility (default is 42).
    - threshold_target_value (float): Quantile value for capping the target (default is 0.9).
    - trans_method (str): The transformation method used ('log', 'boxcox', or 'none').
    - decay_func (bool): If True, applies exponential decay to sample weights.
    - split_test_size (float): Proportion of the dataset to be used as the test set (default is 0.35).

    Returns:
    - model_data (xgb.XGBRegressor): a pickle file contain The best model after hyperparameter tuning and all necessary parameters to deploy the model.
    - metrics (dict): Evaluation metrics including R², MAE, MSE, and RMSE.
    """

    # Step 1: Load the training data from the Excel file
    training_data = pd.read_excel(excel_path, sheet_name='Data')

    # Step 2: Cap 'Incurred' column based on the provided threshold and generate the target column
    cap_incurred_and_drop_columns(training_data, threshold_target_value=threshold_target_value, verbose=verbose)

    # Step 2.1: Dynamically generate the target column name based on the threshold_target_value
    target = f'Incurred_capped_{threshold_target_value * 100}p'

    # Step 2.2: Check if the target column has been generated properly
    if target not in training_data.columns:
        raise ValueError(f"The target column '{target}' is not present in the dataset after capping the 'Incurred' column.")

    # Step 3: Preprocess the data
    preprocess_new_data(training_data, encoding_methods, columns_to_drop=columns_to_drop, 
                        mode='train', verbose=True, target = target)

    # Step 4: Split the data into training and test sets, and apply encoding
    X_train, X_test, y_train, y_test, train_weights, test_weights = prepare_data_for_modeling(
        df=training_data, 
        target=target, 
        weights=weights, 
        encoding_methods=encoding_methods, 
        decay_func=decay_func, 
        split_test_size=split_test_size, 
        verbose=verbose
    )

    # Step 5: Apply transformation to the target variable
    y_train, y_test = apply_transformation(y_train, y_test, trans_method=trans_method, plot=False)

    # Step 6: Define hyperparameter search space if not provided
    if param_dist is None:
        param_dist = {
            'n_estimators': [100, 200, 300, 500, 700], 
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 
            'max_depth': [4, 6, 8, 10, 14, 20], 
            'min_child_weight': [1, 3, 5, 7], 
            'subsample': [0.5, 0.8, 1.0], 
            'colsample_bytree': [0.5, 0.8, 1.0]
        }

    # Step 7: Train the XGBoost model with hyperparameter tuning
    best_xgboost_model, y_pred_transformed = train_xgboost_model(
        X_train=X_train, 
        y_train=y_train, 
        train_weights=train_weights, 
        X_test=X_test, 
        param_dist=param_dist, 
        n_iter=n_iter, 
        scoring=scoring, 
        cv=cv, 
        verbose=verbose, 
        random_state=random_state
    )

    # Step 8: Evaluate the trained model
    metrics, feature_importance_df = evaluate_model_and_save(
        y_test=y_test, 
        y_pred_transformed=y_pred_transformed, 
        best_model=best_xgboost_model, 
        X_train=X_train, 
        trans_method=trans_method,  # Log or other transformation method
        fitted_lambda=None,  # Adjust based on transformation used
        shift_value=0, 
        model_save_path=model_save_path, 
        plot=True
    )

    # Step 9: Save the model and additional parameters to the pickle file
    output_training_data = {
        'feature_names': X_train.columns.tolist(),
        'trans_method': trans_method,
        'fitted_lambda': None,  # No lambda used in log transformation
        'shift_value': 0,  # No shift value for log transformation
        'target': target,  # The target column
        'encoding_methods': encoding_methods,  # Encoding methods used
        'split_test_size': split_test_size,  # Split size used
        'threshold_target_value': threshold_target_value,  # The threshold value for capping
        'columns_to_drop': columns_to_drop  # Columns dropped in preprocessing
    }

    # Save the extended model information in the same pickle file
    with open(training_param_path, 'wb') as f:
        pickle.dump(output_training_data, f)

    print(f"Model and additional parameters saved to {training_param_path}")

    return output_training_data, metrics


####################################################################################################################

if __name__ == "__main__":
    """
    Main execution flow for the project workflow. It handles:
    - Data preprocessing and EDA for the original dataset.
    - Training a machine learning model with optional hyperparameter tuning.
    - Deploying the trained model to new data for making predictions.
    """

    trans_method = 'log'  # type of transformation to be applied to the target to deal with skewnewss of the data option available are: log, Box-Cox, none).
    threshold_target_value = 0.975  # If target is skewed, define a threshold for capping the target (e.g., 99th percentile)
    decay_func= True #if we use exponential decay function for recent and older claims
    split_test_size = 0.33  # This defines the size of the test set as a subsample of the overall dataset. 
    working_path = '/Users/saeideh/project2'
    data_name = '/Users/saeideh/project2/data/Data_Scientist_Interview_Task.xlsx'
    columns_to_drop = ['Claim_number', 'Vechile_registration_present', 'Tp_type_cyclist', 'Tp_type_pedestrian']

    encoding_methods = {
        'Notifier': 'onehot',  
        'Location_of_incident': 'onehot', 
        'Weather_conditions': 'onehot',  
        'Vehicle_mobile': 'onehot',
        'Main_driver': 'onehot',
        'Ph_considered_tp_at_fault': 'onehot'
    }

    param_dist = {
        'n_estimators': [100, 200, 300, 500, 700], 
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 
        'max_depth': [4, 6, 8, 10, 14, 20], 
        'min_child_weight': [1, 3, 5, 7], 
        'subsample': [0.5, 0.8, 1.0], 
        'colsample_bytree': [0.5, 0.8, 1.0]
    }

    best_model, metrics = train_model(
        excel_path='/Users/saeideh/project2/data/Data_Scientist_Interview_Task.xlsx',
        model_save_path='/Users/saeideh/project2/model/best_model.pkl',
        training_param_path='/Users/saeideh/project2/model/training_params.pkl',

        encoding_methods=encoding_methods,
        columns_to_drop=columns_to_drop,
        param_dist=param_dist,
        threshold_target_value=threshold_target_value,
        weights='Weight_days_since_loss',
        decay_func=decay_func,
        split_test_size = split_test_size,
        trans_method=trans_method,
        n_iter=100,
        scoring='neg_mean_absolute_error',
        cv=10,
        random_state=42,
        verbose=2
    )



