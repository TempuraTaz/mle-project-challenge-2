import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn import ensemble
from sklearn import pipeline
from sklearn import preprocessing

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics

# Top 12 features based on V3 feature importance analysis
# These 12 features explain 80% of price variance
TOP_12_FEATURES = [
    'sqft_living',           # 15.3% importance
    'grade',                 # 13.1%
    'sqft_above',            # 8.5%
    'sqft_living15',         # 7.1%
    'bathrooms',             # 6.2%
    'hous_val_amt',          # 5.9% (demographic)
    'medn_incm_per_prsn_amt',# 4.8% (demographic)
    'per_bchlr',             # 4.5% (demographic)
    'per_prfsnl',            # 3.6% (demographic)
    'lat',                   # 3.5%
    'view',                  # 3.4%
    'per_hsd'                # 2.7% (demographic)
]

# House features needed from sales CSV (+ zipcode for merge)
SALES_COLUMN_SELECTION = [
    'price', 'sqft_living', 'grade', 'sqft_above', 'sqft_living15',
    'bathrooms', 'lat', 'view', 'zipcode'
]

OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv("data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")

    # Remove the target variable from the dataframe
    y = merged_data.pop('price')

    # Select only the top 12 features (feature selection based on V3 analysis)
    x = merged_data[TOP_12_FEATURES]

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, test_size=0.25, random_state=42)

    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        ensemble.RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    ).fit(x_train, y_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model_v4.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features_v4.json", 'w'))

    # Calculate and save defaults for missing features
    # Strategy: Use training data medians (robust to outliers)
    # These defaults are used by /predict-minimal when fields are missing
    house_features = ['sqft_living', 'sqft_above', 'sqft_living15', 'bathrooms']

    defaults = {}
    for feature in house_features:
        if feature in x_train.columns:
            median_value = x_train[feature].median()
            # Convert to appropriate type
            if feature in ['sqft_living', 'sqft_above', 'sqft_living15']:
                defaults[feature] = int(median_value)
            else:
                defaults[feature] = float(median_value)

    json.dump(defaults, open(output_dir / "defaults_v4.json", 'w'), indent=2)
    print(f"\nDefaults calculated and saved to {output_dir / 'defaults_v4.json'}")
    print(f"Defaults: {json.dumps(defaults, indent=2)}")
    print(f"\nModel V4: Trained with top 12 features (80% of prediction power)")
    print(f"Features: {TOP_12_FEATURES}")


if __name__ == "__main__":
    main()
