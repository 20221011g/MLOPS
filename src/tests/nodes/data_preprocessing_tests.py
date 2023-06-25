import unittest
import pandas as pd
from src.mlops.pipelines.data_preprocessing.nodes import clean_data

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        # Import DataFrame from a CSV file in a directory
        file_path = 'C:/Users/couto/PycharmProjects/MLOPS/data/01_raw/train.csv'
        self.data = pd.read_csv(file_path)

    def test_clean_data(self):
        cleaned_data, describe_dict, describe_dict_verified = clean_data(self.data)

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
        self.assertIsInstance(describe_dict, dict)
        self.assertIsInstance(describe_dict_verified, dict)

if __name__ == '__main__':
    unittest.main()
