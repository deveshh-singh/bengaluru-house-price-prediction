import unittest
import pandas as pd
import numpy as np
from src.data_preprocessing import float_val, string_val, average_val, clean_size_column, clean_location_column, clean_total_sqft_column, preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def test_float_val(self):
        self.assertEqual(float_val('123.45'), 123.45)
        self.assertEqual(float_val(123), 123.0)
        self.assertTrue(np.isnan(float_val('abc')))
        self.assertTrue(np.isnan(float_val(None)))

    def test_string_val(self):
        self.assertTrue(np.isnan(string_val('123.45')))
        self.assertTrue(np.isnan(string_val(123)))
        self.assertEqual(string_val('abc'), 'abc')
        self.assertEqual(string_val(None), None)
        self.assertEqual(string_val('34.46Sq. Meter'), '34.46Sq. Meter')

    def test_average_val(self):
        self.assertEqual(average_val('10', '20'), 15.0)
        self.assertEqual(average_val(10, 20), 15.0)
        self.assertEqual(average_val('10-20', '30'), '10-20') # v_1 is not a float
        self.assertEqual(average_val('10', 'abc'), 10.0) # v_2 is not a float, but v_1 is
        self.assertEqual(average_val(None, 20), None)
        self.assertEqual(average_val(10, None), 10.0)

    def test_clean_size_column(self):
        data = {'size': ['2 BHK', '4 Bedroom', '3 BHK', np.nan, '1 RK'],
                'total_sqft': [1000, 2000, 1500, 500, 300]}
        df = pd.DataFrame(data)
        cleaned_df = clean_size_column(df.copy())
        self.assertListEqual(list(cleaned_df['size']), [2.0, 4.0, 3.0, 1.0, 1.0])

        data_nan_sqft = {'size': [np.nan], 'total_sqft': [np.nan]}
        df_nan_sqft = pd.DataFrame(data_nan_sqft)
        cleaned_df_nan_sqft = clean_size_column(df_nan_sqft.copy())
        self.assertTrue(np.isnan(cleaned_df_nan_sqft['size'].iloc[0]))

    def test_clean_location_column(self):
        data = {'location': ['Electronic City', 'Kanakpura Road', 'Kothanur', np.nan]}
        df = pd.DataFrame(data)
        location_data = {'browser_link': [
            'Electronic City Bangalore site:codepin.in',
            'Kanakpura Road Bangalore site:codepin.in',
            'Kothanur Bangalore site:codepin.in',
            'Invalid Location Bangalore site:codepin.in'
        ],
            'pincode': [560100, 560099, 560077, np.nan]
        }
        location_df = pd.DataFrame(location_data)
        cleaned_df = clean_location_column(df.copy(), location_df)
        self.assertListEqual(list(cleaned_df['location']), [560100.0, 560099.0, 560077.0, np.nan])

    def test_clean_total_sqft_column(self):
        data = {'total_sqft': ['1000', '1200 - 1500', '500Sq. Meter', 'abc']}
        df = pd.DataFrame(data)
        conversion_data = {'unit': ['SqMeter'], 'conversion_factor': [10.7639]}
        conversion_factor_df = pd.DataFrame(conversion_data)
        cleaned_df = clean_total_sqft_column(df.copy(), conversion_factor_df)
        expected_sqft = [1000.0, 1350.0, 5381.95, np.nan] # 500 * 10.7639 = 5381.95
        pd.testing.assert_series_equal(cleaned_df['total_sqft'], pd.Series(expected_sqft, dtype=object))

    def test_preprocess_data(self):
        # Mock data similar to the notebook's initial state
        data = {
            'area_type': ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Super built-up Area', 'Super built-up Area'],
            'availability': ['19-Dec', 'Ready To Move', 'Ready To Move', 'Ready To Move', 'Ready To Move'],
            'location': ['Electronic City Phase II', 'Chikka Tirupathi', 'Uttarahalli', 'Lingadheeranahalli', 'Kothanur'],
            'size': ['2 BHK', '4 Bedroom', '3 BHK', '3 BHK', '2 BHK'],
            'society': ['Coomee', 'Theanmp', np.nan, 'Soiewre', np.nan],
            'total_sqft': ['1056', '2600', '1440', '1521', '1200'],
            'bath': [2.0, 5.0, 2.0, 3.0, 2.0],
            'balcony': [1.0, 3.0, 3.0, 1.0, 1.0],
            'price': [39.07, 120.00, 62.00, 95.00, 51.00],
            'data_category': ['train', 'train', 'train', 'train', 'train']
        }
        df = pd.DataFrame(data)

        location_data = {'browser_link': [
            'Electronic City Phase II Bangalore site:codepin.in',
            'Kanakpura Road Bangalore site:codepin.in',
            'Kothanur Bangalore site:codepin.in',
            'Invalid Location Bangalore site:codepin.in'
        ],
            'pincode': [560100, 560099, 560077, np.nan]
        }
        location_df = pd.DataFrame(location_data)

        conversion_data = {'unit': ['SqMeter', 'Perch'], 'conversion_factor': [10.7639, 272.25]}
        conversion_factor_df = pd.DataFrame(conversion_data)

        processed_df = preprocess_data(df.copy(), location_df, conversion_factor_df, is_train=True)

        # Assertions for some key transformations
        self.assertIn('avg_price_per_loc', processed_df.columns)
        self.assertTrue(pd.api.types.is_float_dtype(processed_df['total_sqft']))
        self.assertTrue(pd.api.types.is_float_dtype(processed_df['price']))
        self.assertTrue(pd.api.types.is_categorical_dtype(processed_df['area_type']))
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df['availability']))
        self.assertTrue(all(processed_df['society'].isin([0, 1])))
        self.assertTrue(processed_df['total_sqft'].apply(lambda x: isinstance(x, (float, np.floating))).all())
        self.assertTrue(processed_df['price'].apply(lambda x: isinstance(x, (float, np.floating))).all())
        # Check for NaN values introduced in location and total_sqft during preprocessing
        # Some location values will be NaN if not found in location_df, that is expected.
        # self.assertFalse(processed_df['location'].isnull().any())
        self.assertFalse(processed_df['total_sqft'].isnull().any())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
