"""
Feature Importance Analysis

Analyzes which features contribute most to model predictions.
Works with Random Forest models (v2, v3).
"""

import pickle
import json
import pandas as pd
from pathlib import Path
import argparse


def analyze_feature_importance(model_path: str, features_path: str, top_n: int = 20):
    """
    Extract and display feature importance from a trained Random Forest model.

    Args:
        model_path: Path to pickled model
        features_path: Path to JSON file with feature names
        top_n: Number of top features to display
    """
    print("=" * 70)
    print(f"FEATURE IMPORTANCE ANALYSIS: {Path(model_path).stem}")
    print("=" * 70)

    # Load model and features
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    # Extract Random Forest from pipeline
    # Pipeline structure: RobustScaler -> RandomForestRegressor
    rf_model = model.named_steps['randomforestregressor']

    # Get feature importances (Gini importance / Mean Decrease Impurity)
    # This is calculated from the trained trees, not stored separately
    importances = rf_model.feature_importances_

    # Create DataFrame for easy analysis
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Calculate cumulative importance
    importance_df['cumulative'] = importance_df['importance'].cumsum()
    importance_df['cumulative_pct'] = importance_df['cumulative'] / importance_df['importance'].sum() * 100

    # Display top N features
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12} {'Cumulative %':<12}")
    print("-" * 70)

    for idx, row in importance_df.head(top_n).iterrows():
        rank = importance_df.index.get_loc(idx) + 1
        print(f"{rank:<6} {row['feature']:<25} {row['importance']:<12.4f} {row['cumulative_pct']:<12.1f}%")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)

    # How many features explain 80% of importance?
    features_80pct = (importance_df['cumulative_pct'] <= 80).sum()
    print(f"• {features_80pct} features explain 80% of model predictions")
    print(f"• Total features: {len(feature_names)}")

    # Top 5 features
    top_5 = importance_df.head(5)['feature'].tolist()
    print(f"\n• Top 5 drivers: {', '.join(top_5)}")

    # Categorize features
    house_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                      'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                      'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
                      'sqft_living15', 'sqft_lot15']

    importance_df['category'] = importance_df['feature'].apply(
        lambda x: 'House' if x in house_features else 'Demographics'
    )

    category_importance = importance_df.groupby('category')['importance'].sum()
    print(f"\n• Feature category contributions:")
    for cat, imp in category_importance.items():
        print(f"  - {cat}: {imp:.1%}")

    # Low-importance features (potential candidates for removal)
    low_importance = importance_df[importance_df['importance'] < 0.01]
    if len(low_importance) > 0:
        print(f"\n• {len(low_importance)} features have <1% importance")
        print(f"  → Could be removed to simplify model")
        if len(low_importance) <= 10:
            print(f"  Examples: {', '.join(low_importance['feature'].tolist())}")
        else:
            print(f"  Examples: {', '.join(low_importance.head(5)['feature'].tolist())} ...")

    print("=" * 70)
    print("\nHow to interpret:")
    print("  • Importance = % of total prediction power")
    print("  • High (>5%): Strong predictor, keep it")
    print("  • Medium (1-5%): Useful predictor")
    print("  • Low (<1%): Weak/redundant, consider removing")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance for housing models')
    parser.add_argument('--model', type=str, default='model/model_v3.pkl',
                        help='Path to model pickle file (default: model_v3.pkl)')
    parser.add_argument('--features', type=str, default='model/model_features_v3.json',
                        help='Path to features JSON file (default: model_features_v3.json)')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top features to display (default: 20)')

    args = parser.parse_args()

    try:
        analyze_feature_importance(args.model, args.features, args.top)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've trained the model first (run create_model_v3.py)")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
