# Insurance_claim_values_prediction
Develop a model to predict the cost of individual claims using XGBoost algorithm.

To install these dependencies, run:
```
pip install -r requirements.txt
```
## Workflow Overview

1. **Data Preprocessing**: The data is cleaned and processed using `preprocessing.py` or directly in the notebook.
2. **Exploratory Data Analysis**: EDA is performed using `EDA.py` to better understand the dataset and its features.
3. **Model Training**: The model is trained using `training.py`. The best hyperparameters are selected through cross-validation.
4. **Model Deployment**: Once trained, the model can be deployed on new data using `deployment.py`.


## Compatibility
This code is compatible with Python 3.8 and above.


## Usage Instructions

### 1. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Preprocess the Data:
You can preprocess your data by running the `preprocessing.py` file or by calling the appropriate functions from the module in your script.

### 3. Explore the Data:
Run the `EDA.py` to visualize your data and perform exploratory analysis.

### 4. Train the Model:
Train the model by running the `training.py` file. This script will preprocess the data, train the model, and save the trained model to a file.

### 5. Deploy the Model:
Once the model is trained, use the `deployment.py` script to deploy the model and generate predictions on new data.
