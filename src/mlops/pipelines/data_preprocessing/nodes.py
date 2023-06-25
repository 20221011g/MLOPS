import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler


def clean_data(
        data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Does some data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    numerical_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
    categorical_cols = [col for col in data.columns if data[col].dtype not in ['float64', 'int64']]
    # remove columns
    columns_to_drop = ['Id', 'PoolQC', 'Alley', 'MiscFeature', 'Fence', 'FireplaceQu', 'Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating']
    data.drop(columns_to_drop, axis=1, inplace=True)
    #data = data.drop('Id', 'PoolQC', 'Alley', 'MiscFeature', 'Fence', 'FireplaceQu', axis=1, inplace=True)
    #data.drop('Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', axis=1, inplace=True)

    #Fill Null values of numeric features  with knn imputer
    numerical_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
    categorical_cols = [col for col in data.columns if data[col].dtype not in ['float64', 'int64']]

    imputer = KNNImputer(n_neighbors=5)

    data_train_imputed = pd.DataFrame(imputer.fit_transform(data[numerical_cols]), columns=numerical_cols)

    data = pd.concat([data.drop(columns=numerical_cols), data_train_imputed], axis=1)

    # Substituir os valores nulos pela moda em todas as colunas categóricas
    # Loop pelas colunas categóricas
    for column in data.select_dtypes(include='object'):
        mode_value = data[column].mode()[0]  # Calcula a moda da coluna
        data[column].fillna(mode_value, inplace=True)  # Preenche os valores nulos com a moda

    #D some more features seleciton
    eliminator = MultiCollinearityEliminator(df=data, target='SalePrice', threshold=0.80)
    data_cleaned = eliminator.autoEliminateMulticollinearity()


    # Remove outliers based on a specific condition for 'SalePrice'
    data_cleaned = remove_outliers(data_cleaned, 'MSSubClass', 150)
    data_cleaned = remove_outliers(data_cleaned, 'LotFrontage', 200)
    data_cleaned = remove_outliers(data_cleaned, 'LotArea', 100000)
    data_cleaned = remove_outliers(data_cleaned, 'MasVnrArea', 1200)
    data_cleaned = remove_outliers(data_cleaned, 'BsmtFinSF1', 3000)
    data_cleaned = remove_outliers(data_cleaned, 'BsmtFinSF2', 1200)
    data_cleaned = remove_outliers(data_cleaned, 'BsmtUnfSF', 2300)
    data_cleaned = remove_outliers(data_cleaned, 'TotalBsmtSF', 3000)
    data_cleaned = remove_outliers(data_cleaned, 'GrLivArea', 4000)
    data_cleaned = remove_outliers(data_cleaned, 'BsmtFullBath', 2.5)
    data_cleaned = remove_outliers(data_cleaned, 'BedroomAbvGr', 4.5)
    data_cleaned = remove_outliers(data_cleaned, 'WoodDeckSF', 800)
    data_cleaned = remove_outliers(data_cleaned, 'OpenPorchSF', 500)
    data_cleaned = remove_outliers(data_cleaned, 'EnclosedPorch', 350)




    # Criando uma instância do BinaryEncoder
    binary_encoder = ce.BinaryEncoder(cols=categorical_cols)

    # Aplying binary encoding to the df
    data_cleaned = binary_encoder.fit_transform(data_cleaned)

    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    describe_to_dict = data_cleaned.describe().to_dict()

    # Calculate descriptive statistics of the transformed DataFrame
    describe_to_dict_verified = data_cleaned.describe().to_dict()

    data_cleaned.to_csv('C:/Users/couto/PycharmProjects/MLOPS/data/02_intermediate/cleaned_data.csv', index=False)
    return data_cleaned, describe_to_dict, describe_to_dict_verified


def feature_engineer(
        cleaned_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """Does sfeature engineering and selection.
    Args:
        cleaned_data: Data containing features and target.
    Returns:
        data_engineered: Data after feature engineering
    """
    # Turn 'BsmtFinSF2' into a binary feature
    cleaned_data['BsmtFinSF2'] = (cleaned_data['BsmtFinSF2'] > 0).astype(int)

    # Drop rows with NaN values
    cleaned_data.dropna(inplace=True)

    # Drop target from dataset
    cleaned_data_no_target = cleaned_data.drop(labels=['target'], axis=1)

    # create features and target
    X = cleaned_data_no_target
    y = cleaned_data['target']

    # convert to categorical data by converting data to integers
    X = X.astype(int)

    # Compute the mutual information between each feature and the target variable
    mi_scores = mutual_info_regression(X, y)

    # Create a dictionary to store feature scores
    feature_scores = dict(zip(X.columns, mi_scores))

    # Filter features with scores greater than 0
    best_info_columns = [feature for feature, score in feature_scores.items() if score > 0]

    # Make dataframe with the best columns
    X = X[best_info_columns]

    # using sklearn variancethreshold to find nearly-constant features
    sel = VarianceThreshold(threshold=0.01)
    sel.fit(X)  # fit finds the features with very small variance

    # Get the indices of included features
    best_columns = X.columns[sel.get_support()]

    # Transform X to include only the selected features
    X = sel.transform(X)

    # Convert X back to a DataFrame with column names
    data_engineered = pd.DataFrame(X, columns=best_columns)

    # Calculate descriptive statistics of the transformed DataFrame
    describe_to_dict_verified = data_engineered.describe().to_dict()

    return data_engineered, describe_to_dict_verified


def remove_outliers(data, col, val):
    for index, value in data[col].items():
        if value > val:
            data.drop(index, inplace=True)

    return data

#Feature selection class to eliminate multicollinearity
class MultiCollinearityEliminator():

    #Class Constructor
    def __init__(self, df, target, threshold):
        self.df = df
        self.target = target
        self.threshold = threshold

    #Method to create and return the feature correlation matrix dataframe
    def createCorrMatrix(self, include_target = False):
        #Checking we should include the target in the correlation matrix
        if (include_target == False):
            df_temp = self.df.drop([self.target], axis =1)

            #Setting method to Pearson to prevent issues in case the default method for df.corr() gets changed
            #Setting min_period to 30 for the sample size to be statistically significant (normal) according to
            #central limit theorem
            corrMatrix = df_temp.corr(method='pearson', min_periods=30).abs()
        #Target is included for creating the series of feature to target correlation - Please refer the notes under the
        #print statement to understand why we create the series of feature to target correlation
        elif (include_target == True):
            corrMatrix = self.df.corr(method='pearson', min_periods=30).abs()
        return corrMatrix

    #Method to create and return the feature to target correlation matrix dataframe
    def createCorrMatrixWithTarget(self):
        #After obtaining the list of correlated features, this method will help to view which variables
        #(in the list of correlated features) are least correlated with the target
        #This way, out the list of correlated features, we can ensure to elimate the feature that is
        #least correlated with the target
        #This not only helps to sustain the predictive power of the model but also helps in reducing model complexity

        #Obtaining the correlation matrix of the dataframe (along with the target)
        corrMatrix = self.createCorrMatrix(include_target = True)
        #Creating the required dataframe, then dropping the target row
        #and sorting by the value of correlation with target (in asceding order)
        corrWithTarget = pd.DataFrame(corrMatrix.loc[:,self.target]).drop([self.target], axis = 0).sort_values(by = self.target)
        #print(corrWithTarget, '\n')
        return corrWithTarget

    #Method to create and return the list of correlated features
    def createCorrelatedFeaturesList(self):
        #Obtaining the correlation matrix of the dataframe (without the target)
        corrMatrix = self.createCorrMatrix(include_target = False)
        colCorr = []
        #Iterating through the columns of the correlation matrix dataframe
        for column in corrMatrix.columns:
            #Iterating through the values (row wise) of the correlation matrix dataframe
            for idx, row in corrMatrix.iterrows():
                if(row[column]>self.threshold) and (row[column]<1):
                    #Adding the features that are not already in the list of correlated features
                    if (idx not in colCorr):
                        colCorr.append(idx)
                    if (column not in colCorr):
                        colCorr.append(column)
        # print(colCorr, '\n')
        return colCorr

    #Method to eliminate the least important features from the list of correlated features
    def deleteFeatures(self, colCorr):
        #Obtaining the feature to target correlation matrix dataframe
        corrWithTarget = self.createCorrMatrixWithTarget()
        for idx, row in corrWithTarget.iterrows():
            #print(idx, '\n')
            if (idx in colCorr):
                self.df = self.df.drop(idx, axis =1)
                break
        return self.df

    #Method to run automatically eliminate multicollinearity
    def autoEliminateMulticollinearity(self):
        #Obtaining the list of correlated features
        colCorr = self.createCorrelatedFeaturesList()
        while colCorr != []:
            #Obtaining the dataframe after deleting the feature (from the list of correlated features)
            #that is least correlated with the taregt
            self.df = self.deleteFeatures(colCorr)
            #Obtaining the list of correlated features
            colCorr = self.createCorrelatedFeaturesList()
        return self.df