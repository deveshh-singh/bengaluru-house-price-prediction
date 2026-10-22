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
        
        # New test cases for clean_size_column
        # Test case: size is NaN, total_sqft is 0
        data_zero_sqft = {'size': [np.nan], 'total_sqft': [0]}
        df_zero_sqft = pd.DataFrame(data_zero_sqft)
        cleaned_df_zero_sqft = clean_size_column(df_zero_sqft.copy())
        self.assertEqual(cleaned_df_zero_sqft['size'].iloc[0], 0.0) # ceil(0/550) = 0
        
        # Test case: size is NaN, total_sqft is a small number (e.g., 100)
        data_small_sqft = {'size': [np.nan], 'total_sqft': [100]}
        df_small_sqft = pd.DataFrame(data_small_sqft)
        cleaned_df_small_sqft = clean_size_column(df_small_sqft.copy())
        self.assertEqual(cleaned_df_small_sqft['size'].iloc[0], 1.0) # ceil(100/550) = 1

        # Test case: size is NaN, total_sqft is a large number (e.g., 3000)
        data_large_sqft = {'size': [np.nan], 'total_sqft': [3000]}
        df_large_sqft = pd.DataFrame(data_large_sqft)
        cleaned_df_large_sqft = clean_size_column(df_large_sqft.copy())
        self.assertEqual(cleaned_df_large_sqft['size'].iloc[0], 6.0) # ceil(3000/550) = 6
        
        # Test case: mixed valid and NaN sizes with total_sqft for imputation
        data_mixed = {'size': ['2 BHK', np.nan, '1 RK'], 'total_sqft': [1000, 700, 300]}
        df_mixed = pd.DataFrame(data_mixed)
        cleaned_df_mixed = clean_size_column(df_mixed.copy())
        self.assertListEqual(list(cleaned_df_mixed['size']), [2.0, 2.0, 1.0]) # ceil(700/550) = 2


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
        data = {'total_sqft': ['1000', '1200 - 1500', '500Sq. Meter', 'abc', '2000 Sq. Yards', '3000 Cents', '1000-1200']}
        df = pd.DataFrame(data)
        conversion_data = {'unit': ['SqMeter', 'Sq. Yards', 'Cents'], 'conversion_factor': [10.7639, 9.0, 435.6]}
        conversion_factor_df = pd.DataFrame(conversion_data)
        cleaned_df = clean_total_sqft_column(df.copy(), conversion_factor_df)
        expected_sqft = [1000.0, 1350.0, 5381.95, np.nan, 18000.0, 1306800.0, 1100.0] # 500 * 10.7639 = 5381.95; 2000 * 9 = 18000; 3000 * 435.6 = 1306800
        pd.testing.assert_series_equal(cleaned_df['total_sqft'], pd.Series(expected_sqft, dtype=object))

        # Test with empty conversion_factor_df
        df_empty_conv = pd.DataFrame({'total_sqft': ['100 Sq. Meter']})
        empty_conversion_df = pd.DataFrame(columns=['unit', 'conversion_factor'])
        cleaned_df_empty_conv = clean_total_sqft_column(df_empty_conv.copy(), empty_conversion_df)
        # If unit not found, it should default to using the numeric part as is (conversion factor 1)
        self.assertEqual(cleaned_df_empty_conv['total_sqft'].iloc[0], 100.0) 

        # Test with no matching units in conversion_factor_df
        df_no_match_conv = pd.DataFrame({'total_sqft': ['100 Furlongs']})
        no_match_conversion_df = pd.DataFrame({'unit': ['SqMeter'], 'conversion_factor': [10.7639]})
        cleaned_df_no_match_conv = clean_total_sqft_column(df_no_match_conv.copy(), no_match_conversion_df)
        self.assertEqual(cleaned_df_no_match_conv['total_sqft'].iloc[0], 100.0)

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
        
        # Additional assertions for society, availability, bath, balcony, and log transformation
        self.assertListEqual(list(processed_df['society']), [1, 1, 0, 1, 0]) # society NaN should be 0, others 1
        self.assertListEqual(list(processed_df['availability']), [0, 1, 1, 1, 1]) # Ready To Move -> 1, others -> 0
        
        # Imputed bath: original [2.0, 5.0, 2.0, 3.0, 2.0] - no NaNs, should be unchanged
        self.assertListEqual(list(processed_df['bath']), [2.0, 5.0, 2.0, 3.0, 2.0])
        
        # Balcony imputation: original [1.0, 3.0, 3.0, 1.0, 1.0] - no NaNs, should be unchanged
        self.assertListEqual(list(processed_df['balcony']), [1.0, 3.0, 3.0, 1.0, 1.0])
        
        # Check log transformation for total_sqft and price (only if is_train=True)
        # We can't directly check the value without recalculating, but we can check if they are positive and float
        self.assertTrue((processed_df['total_sqft'] > 0).all())
        self.assertTrue((processed_df['price'] > 0).all())

    def test_preprocess_data_is_not_train(self):
        data = {
            'area_type': ['Super built-up Area', 'Plot Area'],
            'availability': ['19-Dec', 'Ready To Move'],
            'location': ['Electronic City Phase II', 'Chikka Tirupathi'],
            'size': ['2 BHK', np.nan],
            'society': ['Coomee', np.nan],
            'total_sqft': ['1056', '2600'],
            'bath': [2.0, np.nan],
            'balcony': [1.0, np.nan],
            'price': [39.07, 120.00],
            'data_category': ['test', 'test'] # Mark as test data
        }
        df = pd.DataFrame(data)

        location_data = {'browser_link': [
            'Electronic City Phase II Bangalore site:codepin.in',
            'Kanakpura Road Bangalore site:codepin.in'
        ],
            'pincode': [560100, 560099]
        }
        location_df = pd.DataFrame(location_data)

        conversion_data = {'unit': ['SqMeter'], 'conversion_factor': [10.7639]}
        conversion_factor_df = pd.DataFrame(conversion_data)

        processed_df = preprocess_data(df.copy(), location_df, conversion_factor_df, is_train=False)

        # 'avg_price_per_loc' should not be in columns when is_train is False
        self.assertNotIn('avg_price_per_loc', processed_df.columns)
        
        # price should not be log transformed
        # check if it's still the original values (approximately)
        self.assertListEqual(list(processed_df['price']), [39.07, 120.00])

        # Check other transformations that should still apply
        self.assertListEqual(list(processed_df['society']), [1, 0])
        self.assertListEqual(list(processed_df['availability']), [0, 1])
        
        # Bath imputation: size for second row is 4.0 (ceil(2600/550)=5), so bath becomes 5.0
        # Original bath: [2.0, np.nan] -> expected [2.0, 5.0] (2600/550 is 4.72, so 5)
        self.assertListEqual(list(processed_df['bath']), [2.0, 5.0])

        # Balcony imputation: median of balcony based on size. For size=5, median might be 4 if not enough data.
        # Let's assume, for now, that the imputation fills with a reasonable value, 
        # or verify the actual logic. Based on the code, if balcony is NaN it will map from balcony_dict.
        # balcony_dict gets its values from grouping by 'size' from the input df.
        # Given our current input df, for size = 2 (row 0), balcony is 1.0.
        # For size = 5 (row 1, after size imputation), there is no '5' in the input df's 'size' to derive median balcony.
        # The code will apply balcony_dict, and if no mapping is found, it will remain NaN. If balcony_dict has a default for NaNs it will apply.
        # From original code: balcony['balcony'].fillna(4, inplace=True). So if size is not in balcony_dict, it fills with 4.
        # After clean_size_column, first row size is 2.0, second row size is 5.0.
        # So, first row balcony remains 1.0. Second row balcony (was NaN) becomes 4.0 due to fillna(4) in `balcony` processing.
        self.assertListEqual(list(processed_df['balcony']), [1.0, 4.0])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)