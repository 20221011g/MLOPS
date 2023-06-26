
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler

def clean_data(
        data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    if 'SalePrice' in data.columns:
        # Remove outliers based on a specific condition for 'SalePrice'
        data = remove_outliers(data, 'SalePrice', 700000)
        # Rename 'SalePrice' column to 'target'
        data = data.rename(columns={'SalePrice': 'target'})

    # Remove outliers for other columns
    data = remove_outliers(data, 'TotalBsmtSF', 5000)
    data = remove_outliers(data, 'LotArea', 100000)

    # Create a copy of the DataFrame for further cleaning
    df_transformed = data.copy()
    describe_to_dict = pd.DataFrame(df_transformed.describe().to_dict())

    # Drop rows with missing values in columns other than 'target'
    data.dropna(subset=df_transformed.columns[df_transformed.columns != 'target'], inplace=True)

    if 'target' in df_transformed.columns:
        # Impute missing values in 'target' column using the mean strategy
        imputer = SimpleImputer(strategy='mean')
        target = df_transformed['target'].values.reshape(-1, 1)
        imputer.fit(target)
        df_transformed['target'] = imputer.transform(target)

    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    describe_to_dict = data_cleaned.describe().to_dict()

    # Calculate descriptive statistics of the transformed DataFrame
    describe_to_dict_verified = pd.DataFrame(df_transformed.describe().to_dict())



def feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    # delete null columns and rows with null values
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='any')

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

    # Ensure all other columns are of a numeric type, else convert them
    for col in data.columns:
        if data[col].dtype not in ['int64', 'float64']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data

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

