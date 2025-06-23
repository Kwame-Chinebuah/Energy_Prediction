
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def preprocess_consumption_data(df,
                                 gdp_thresh_fixed=13448.84,
                                 pop_thresh_fixed=373749.49,
                                 mapping_file='local_authority_mapping.csv',
                                 encoders_path='label_encoders.pkl',
                                 training=True):

    # Load local authority mapping
    mapping_df = pd.read_csv(mapping_file)

    # Merge mapping into input DataFrame without overwriting Region/Weather_zone
    df = df.merge(mapping_df, on='Local authority', how='left', suffixes=('', '_drop'))
    df = df.drop(columns=[col for col in df.columns if col.endswith('_drop')])

    # Fill missing region or weather zone with placeholders if needed
    df['Region'] = df['Region'].fillna('Unknown')
    df['Weather_zone'] = df['Weather_zone'].fillna('Unknown')

    # Log-transform GDP and Population to reduce skewness
    df['log_GDP'] = np.log1p(df['GDP']).round(6)
    df['log_Population'] = np.log1p(df['Population']).round(6)


    # Use fixed thresholds for high GDP and population flags
    df['high_GDP'] = (df['GDP'] > gdp_thresh_fixed).astype(int)
    df['high_population'] = (df['Population'] > pop_thresh_fixed).astype(int)

    categorical_columns = ['Local authority', 'Region', 'Weather_zone', 'high_GDP', 'high_population']

    if training:
        # Fit new encoders
        encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        # Save encoders for later use
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
    else:
        # Load encoders and transform
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders file '{encoders_path}' not found. You need to train first.")

        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)

        for col in categorical_columns:
            df[col] = encoders[col].transform(df[col])

    # Select relevant columns
    cols_to_keep = [
        'Region', 'Local authority', 'Total_meters(K)', 'Year',
        'Weather_zone', 'Avg_rainfall', 'Avg_mean_temp',
        'log_GDP', 'log_Population', 'high_GDP', 'high_population', 'GDP_per_capita'
    ]
    df_transformed = df[cols_to_keep].copy()

    return df_transformed, encoders

