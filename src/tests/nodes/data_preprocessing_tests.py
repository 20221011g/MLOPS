import unittest
import pandas as pd
from src.mlops.pipelines.data_preprocessing.nodes import clean_data
from src.mlops.pipelines.data_preprocessing.nodes import feature_engineer


class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        # Import DataFrame from a CSV file in a directory
        file_path = 'C:/Users/couto/PycharmProjects/MLOPS/data/01_raw/train.csv'
        self.data = pd.read_csv(file_path)
        self.cleaned_data = None  # Initialize cleaned_data as an instance variable


    def test_clean_data(self):
        cleaned_data, raw_describe, cleaned_describe = clean_data(self.data)
        self.cleaned_data = cleaned_data  # Assign cleaned_data to the instance variable

        # Assertions for dropped columns
        self.assertNotIn('Id', cleaned_data.columns)
        self.assertNotIn('PoolQC', cleaned_data.columns)
        self.assertNotIn('Alley', cleaned_data.columns)
        self.assertNotIn('MiscFeature', cleaned_data.columns)
        self.assertNotIn('Fence', cleaned_data.columns)
        self.assertNotIn('FireplaceQu', cleaned_data.columns)
        self.assertNotIn('Street', cleaned_data.columns)
        self.assertNotIn('Utilities', cleaned_data.columns)
        self.assertNotIn('Condition2', cleaned_data.columns)
        self.assertNotIn('RoofMatl', cleaned_data.columns)
        self.assertNotIn('Heating', cleaned_data.columns)

        self.assertFalse(cleaned_data.isnull().any().any())

        #Expected shape after the final change
        expected_encoded_shape = (1312, 136)
        self.assertEqual(cleaned_data.shape, expected_encoded_shape)  # Update the expected shape after encoding

        # Assertions for describe dictionary
        self.assertIsInstance(raw_describe, dict)
        self.assertIsInstance(cleaned_describe, dict)

class TestFeatureEngineering(unittest.TestCase):
    def test_data_engineer(self):
        # Access the cleaned_data from the previous test
        cleaned_data = TestDataCleaning.cleaned_data

        # Call the data_engineer function with the cleaned_data
        data_engineered, engineered_describe = feature_engineer(cleaned_data)

        #Expected shape after the final change
        expected_preprocessed_shape = (1312, 70)
        self.assertEqual(data_engineered.shape, expected_preprocessed_shape)  # Update the expected shape after encoding

        # Assertions for describe dictionary
        self.assertIsInstance(engineered_describe, dict)

if __name__ == '__main__':
    unittest.main()
