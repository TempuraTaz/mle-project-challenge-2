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
# List of columns (ALL available features) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15',
    'zipcode'
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
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

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
    pickle.dump(model, open(output_dir / "model_v3.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features_v3.json", 'w'))

    # Calculate and save defaults for missing features
    # Strategy: Use training data medians (robust to outliers)
    # These defaults are used by /predict-minimal when fields are missing
    house_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                      'floors', 'sqft_above', 'sqft_basement']

    defaults = {}
    for feature in house_features:
        if feature in x_train.columns:
            median_value = x_train[feature].median()
            # Convert to appropriate type
            if feature in ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']:
                defaults[feature] = int(median_value)
            else:
                defaults[feature] = float(median_value)

    json.dump(defaults, open(output_dir / "defaults_v3.json", 'w'), indent=2)
    print(f"\nDefaults calculated and saved to {output_dir / 'defaults_v3.json'}")
    print(f"Defaults: {json.dumps(defaults, indent=2)}")


if __name__ == "__main__":
    main()
