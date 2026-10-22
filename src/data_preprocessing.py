
import numpy as np
import pandas as pd
import re
import math

def float_val(x):
    """
    Attempts to convert a value to float. If it fails, returns NaN.
    """
    try:
        x = float(x)
    except:
        x = np.nan
    return x

def string_val(x):
    """
    Attempts to convert a value to float. If it succeeds, returns NaN (as it's meant to extract strings).
    If it fails, returns the original value (assumed to be a string).
    """
    try:
        x = float(x)
        if isinstance(x, float):
            x = np.nan
    except:
        pass
    return x

def average_val(v_1, v_2):
    """
    Calculates the average of two values after attempting to convert them to float.
    If conversion fails for v_1, it returns v_1 as is.
    """
    try:
        v_1 = float(v_1)
        v_2 = float(v_2)
        avg = (v_1 + v_2) / 2
    except:
        avg = v_1
    return avg

def clean_size_column(df):
    """
    Cleans the 'size' column by extracting the numeric bedroom count and converting to float.
    Imputes missing 'size' values based on 'total_sqft'.
    """
    df['size'] = df['size'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else np.nan)
    df['size'] = df['size'].astype(float)
    df['size'] = df.apply(lambda row: math.ceil(float(row['total_sqft'] / 550)) if np.isnan(row['size']) else row['size'], axis=1)
    return df

def clean_location_column(df, location_df):
    """
    Cleans the 'location' column by mapping locations to pincodes using a provided DataFrame.
    """
    uniq_locn = pd.DataFrame(df['location'].unique(), columns=['location'])
    uniq_locn['browser_link'] = uniq_locn['location'].apply(lambda x: str(x) + " Bangalore site:codepin.in")
    uniq_locn['pincode'] = (uniq_locn['browser_link'].map(dict(zip(location_df['browser_link'], location_df['pincode']))))
    df['location'] = df['location'].map(dict(zip(uniq_locn['location'], uniq_locn['pincode'])))
    return df

def clean_total_sqft_column(df, conversion_factor):
    """
    Cleans the 'total_sqft' column by handling ranges and different units,
    converting all values to square feet.
    """
    sqft = df[['total_sqft']].copy()
    sqft['float_val'] = sqft['total_sqft'].apply(float_val)
    sqft['string_val'] = sqft['total_sqft'].apply(string_val)

    sqft_string = sqft[sqft['string_val'].notnull()].copy()
    sqft_string['v_1'] = sqft_string['string_val'].apply(lambda x: str(x).split("-")[0])
    sqft_string['v_2'] = sqft_string['string_val'].apply(lambda x: str(x).split("-")[-1])
    sqft_string['avg_val'] = sqft_string.apply(lambda x: average_val(x.v_1, x.v_2), axis=1)

    sqft_string['conversion'] = sqft_string['avg_val'].apply(string_val)
    sqft_string['con_val'] = sqft_string['conversion'].apply(lambda x: "".join(re.findall(r"[a-zA-Z_]", str(x))))
    sqft_string['num_val'] = sqft_string['conversion'].apply(lambda x: ".".join(re.findall("\d+", str(x))))
    
    sqft_string['num_val'] = sqft_string['num_val'].replace('', np.nan).fillna(0).astype(float)
    sqft_string['num_val'] = np.where(sqft_string['num_val'] == 0, sqft_string['avg_val'], sqft_string['num_val'])

    sqft_string['conversion_factor'] = (sqft_string['con_val'].map(dict(zip(conversion_factor['unit'], conversion_factor['conversion_factor']))).fillna(1))
    sqft_string['after_conversion'] = sqft_string['num_val'] * sqft_string['conversion_factor']
    sqft_string['after_conversion'] = sqft_string['after_conversion'].apply(lambda x: round(x, 2))

    string_val_dict = dict(zip(sqft_string['total_sqft'], sqft_string['after_conversion']))
    df['total_sqft_new'] = df['total_sqft'].map(string_val_dict)
    df['total_sqft'] = (df.apply(lambda row: row['total_sqft'] if np.isnan(row['total_sqft_new']) else row['total_sqft_new'], axis=1))
    df['total_sqft'] = df['total_sqft'].astype(float)
    df.drop(['total_sqft_new'], axis=1, inplace=True)
    return df

def preprocess_data(df, location_df, conversion_factor, is_train=True):
    """
    Orchestrates the data preprocessing steps.
    """
    # Handle 'size' column
    df = clean_size_column(df)

    # Handle 'location' column
    df = clean_location_column(df, location_df)

    # Handle 'total_sqft' column
    df = clean_total_sqft_column(df, conversion_factor)

    # Handle 'society' column
    df.loc[(df['society'].isnull()), 'society'] = 0
    df.loc[(df['society'] != 0), 'society'] = 1

    # Handle 'availability' column
    df['availability'].replace(["Ready To Move", "Immediate Possession"], 1, inplace=True)
    df.loc[(df['availability'] != 1), 'availability'] = 0

    # Impute 'bath' column
    df['bath'] = df.apply(lambda row: row['size'] if np.isnan(row['bath']) else row['bath'], axis=1)

    # Impute 'balcony' column
    balcony = df.groupby(['size']).agg({'balcony': 'median'}).reset_index()
    balcony['balcony'].fillna(4, inplace=True)
    balcony_dict = dict(zip(balcony['size'], balcony['balcony']))
    df['new_balcony'] = df[df['balcony'].isna()]['size'].map(balcony_dict)
    df['balcony'] = (df.apply(lambda row: row['new_balcony'] if np.isnan(row['balcony']) else row['balcony'], axis=1))
    df.drop(['new_balcony'], axis=1, inplace=True)
    
    # Deriving new feature 'avg_price_per_loc' if it's the training data
    if is_train:
        price_per_loc = (df[df['data_category'] == 'train'].groupby(['area_type', 'location', 'society'])
                         .agg({'total_sqft': 'sum', 'price': 'sum'}).reset_index())
        price_per_loc['avg_price_per_loc'] = price_per_loc['price'] / price_per_loc['total_sqft']
        df = pd.merge(df, price_per_loc[['area_type', 'location', 'society', 'avg_price_per_loc']],
                      on=['area_type', 'location', 'society'], how='left')

        price_per_loc_2 = (df[df['data_category'] == 'train'].groupby(['location', 'society'])
                           .agg({'total_sqft': 'sum', 'price': 'sum'}).reset_index())
        price_per_loc_2['avg_price_per_loc_2'] = price_per_loc_2['price'] / price_per_loc_2['total_sqft']
        df = pd.merge(df, price_per_loc_2[['location', 'society', 'avg_price_per_loc_2']],
                      on=['location', 'society'], how='left')
        
        df['avg_price_per_loc'] = df.apply(lambda row: row['avg_price_per_loc_2'] if np.isnan(row['avg_price_per_loc']) else row['avg_price_per_loc'], axis=1)

        average_price = df[df['data_category'] == 'train']['price'].sum() / df[df['data_category'] == 'train']['total_sqft'].sum()
        df['avg_price_per_loc'].fillna(average_price, inplace=True)
        df.drop(['location', 'avg_price_per_loc_2'], axis=1, inplace=True)

    # Type conversion
    df['area_type'] = pd.Categorical(df['area_type'])
    df[['size', 'total_sqft', 'bath', 'balcony', 'avg_price_per_loc']] = df[['size', 'total_sqft', 'bath', 'balcony', 'avg_price_per_loc']].apply(pd.to_numeric)
    df[['availability', 'society']] = df[['availability', 'society']].astype(int)

    # Log transformation
    df['total_sqft'] = np.log(df['total_sqft'])
    if is_train:
        df['price'] = np.log(df['price'])
    
    return df

