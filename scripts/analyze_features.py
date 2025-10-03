"""
Comprehensive Feature Analysis Script for King County House Sales Dataset

This script provides detailed analysis of all features including:
- Value ranges and distributions for ALL features
- Data quality checks
- Feature correlations with target variable
- Missing value analysis
- Statistical summaries

Usage:
    python scripts/analyze_features.py

    # Show only top 10 correlations
    python scripts/analyze_features.py --top-corr 10

    # Brief summary only
    python scripts/analyze_features.py --summary-only
"""

import pandas as pd
import numpy as np
import argparse


class FeatureAnalyzer:
    """Analyzes all features in the house sales dataset"""

    def __init__(self, sales_path: str, demographics_path: str):
        """
        Initialize the analyzer with data paths.

        Args:
            sales_path: Path to house sales CSV
            demographics_path: Path to demographics CSV
        """
        self.sales = pd.read_csv(sales_path, dtype={'zipcode': str})
        self.demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})
        self.merged = self.sales.merge(self.demographics, on='zipcode', how='left')

    def print_header(self, text: str, char: str = "="):
        """Print a formatted header"""
        print(f"\n{char * 80}")
        print(text)
        print(f"{char * 80}\n")

    def analyze_numeric_feature(self, col: str, description: str, unit: str = "", source: str = "house"):
        """
        Analyze a numeric feature and print statistics.

        Args:
            col: Column name
            description: Feature description
            unit: Unit of measurement ($, sqft, %, etc.)
            source: "house" or "demographic"
        """
        if source == "house":
            data = self.sales[col] if col in self.sales.columns else None
        else:
            data = self.demographics[col] if col in self.demographics.columns else None

        if data is None:
            return

        # Basic statistics
        min_val = data.min()
        max_val = data.max()
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)

        # Missing values
        missing = data.isna().sum()
        missing_pct = (missing / len(data)) * 100

        # Unique values
        unique = data.nunique()

        print(f"\n{col.upper()}")
        print(f"  Description: {description}")

        # Format based on unit
        if unit == '$':
            print(f"  Range: ${min_val:,.0f} to ${max_val:,.0f}")
            print(f"  Mean: ${mean_val:,.0f} | Median: ${median_val:,.0f}")
            print(f"  Std Dev: ${std_val:,.0f}")
            print(f"  Quartiles: Q1=${q25:,.0f}, Q3=${q75:,.0f}")
        elif unit == '%':
            print(f"  Range: {min_val:.1f}% to {max_val:.1f}%")
            print(f"  Mean: {mean_val:.1f}% | Median: {median_val:.1f}%")
            print(f"  Std Dev: {std_val:.1f}%")
        elif unit in ['sqft', 'count', 'year', 'rating']:
            print(f"  Range: {min_val:,.0f} to {max_val:,.0f}")
            print(f"  Mean: {mean_val:,.1f} | Median: {median_val:,.0f}")
            print(f"  Std Dev: {std_val:,.1f}")
        elif unit == 'geo':
            print(f"  Range: {min_val:.6f} to {max_val:.6f}")
            print(f"  Mean: {mean_val:.6f} | Median: {median_val:.6f}")
        else:
            print(f"  Range: {min_val} to {max_val}")
            print(f"  Mean: {mean_val:.2f} | Median: {median_val:.2f}")

        print(f"  Unique values: {unique:,}")

        if missing > 0:
            print(f"  ⚠ Missing: {missing:,} ({missing_pct:.1f}%)")

        # Special insights
        if unit == 'binary':
            pct_ones = (data == 1).sum() / len(data) * 100
            print(f"  Distribution: {pct_ones:.1f}% are 1 (True), {100-pct_ones:.1f}% are 0 (False)")

        if col == 'bedrooms' and max_val > 10:
            outlier_count = (data > 10).sum()
            print(f"  ⚠ Outlier: {outlier_count} records with >10 bedrooms (max: {max_val:.0f})")

        if col == 'sqft_basement' and median_val == 0:
            pct_zero = (data == 0).sum() / len(data) * 100
            print(f"  Note: {pct_zero:.1f}% have no basement (value = 0)")

        if col == 'yr_renovated' and median_val == 0:
            pct_zero = (data == 0).sum() / len(data) * 100
            print(f"  Note: {pct_zero:.1f}% never renovated (value = 0)")

        if col == 'view' and median_val == 0:
            pct_zero = (data == 0).sum() / len(data) * 100
            print(f"  Note: {pct_zero:.1f}% have no view (value = 0)")

    def analyze_categorical_feature(self, col: str, description: str):
        """Analyze a categorical feature"""
        data = self.sales[col]

        unique = data.nunique()
        missing = data.isna().sum()
        missing_pct = (missing / len(data)) * 100

        print(f"\n{col.upper()}")
        print(f"  Description: {description}")
        print(f"  Unique values: {unique:,}")

        if missing > 0:
            print(f"  ⚠ Missing: {missing:,} ({missing_pct:.1f}%)")

        # Show top 5 most common values
        top_values = data.value_counts().head(5)
        print(f"  Most common:")
        for val, count in top_values.items():
            pct = (count / len(data)) * 100
            print(f"    {val}: {count:,} ({pct:.1f}%)")

    def analyze_house_features(self):
        """Analyze all house features"""
        self.print_header("HOUSE FEATURES ANALYSIS")
        print(f"Dataset: {len(self.sales):,} house sales\n")

        # Define ALL features with metadata
        features = [
            ('price', 'Sale price (TARGET VARIABLE)', '$'),
            ('bedrooms', 'Number of bedrooms', 'count'),
            ('bathrooms', 'Number of bathrooms (can be fractional: .25=quarter, .5=half, .75=three-quarter)', 'count'),
            ('sqft_living', 'Square footage of interior living space', 'sqft'),
            ('sqft_lot', 'Square footage of the land lot', 'sqft'),
            ('floors', 'Number of floors (levels) in house', 'count'),
            ('waterfront', 'Waterfront property (0=No, 1=Yes)', 'binary'),
            ('view', 'View quality rating (0=No view, 1=Fair, 2=Average, 3=Good, 4=Excellent)', 'rating'),
            ('condition', 'Overall condition (1=Poor, 2=Fair, 3=Average, 4=Good, 5=Very Good)', 'rating'),
            ('grade', 'Construction quality (1-3=Falls short, 4-6=Average, 7-10=Good, 11-13=Luxury)', 'rating'),
            ('sqft_above', 'Square footage above ground level', 'sqft'),
            ('sqft_basement', 'Square footage of basement (0 if no basement)', 'sqft'),
            ('yr_built', 'Year house was originally built', 'year'),
            ('yr_renovated', 'Year of most recent renovation (0 if never renovated)', 'year'),
            ('lat', 'Latitude coordinate (geographic location)', 'geo'),
            ('long', 'Longitude coordinate (geographic location)', 'geo'),
            ('sqft_living15', 'Average sqft living space of 15 nearest neighbors (neighborhood proxy)', 'sqft'),
            ('sqft_lot15', 'Average sqft lot size of 15 nearest neighbors (neighborhood proxy)', 'sqft')
        ]

        for col, desc, unit in features:
            if col in self.sales.columns:
                self.analyze_numeric_feature(col, desc, unit, source="house")

        # Categorical features
        self.print_header("CATEGORICAL HOUSE FEATURES", "-")
        self.analyze_categorical_feature('zipcode', 'ZIP code (5-digit postal code, 70 unique zipcodes in King County)')

        # Date analysis
        print(f"\nDATE")
        print(f"  Description: Date house was sold")
        print(f"  Range: {self.sales['date'].min()} to {self.sales['date'].max()}")
        print(f"  Format: YYYYMMDD")

    def analyze_demographic_features(self):
        """Analyze ALL demographic features"""
        self.print_header("DEMOGRAPHIC FEATURES ANALYSIS (U.S. Census Data)")
        print(f"Dataset: {len(self.demographics)} unique zipcodes")
        print("Source: U.S. Census Bureau")
        print("Note: These features are joined to house data via zipcode\n")

        # ALL demographic features with detailed descriptions
        demo_features = [
            # Population
            ('ppltn_qty', 'Total population in zipcode area', 'count'),
            ('urbn_ppltn_qty', 'Urban population count (vs rural)', 'count'),
            ('medn_age_amt', 'Median age of residents', 'count'),
            ('male_qty', 'Number of male residents', 'count'),
            ('female_qty', 'Number of female residents', 'count'),

            # Income & Housing Value
            ('medn_hshld_incm_amt', 'Median household income (all earners in household)', '$'),
            ('medn_incm_per_prsn_amt', 'Median income per person (individual earners)', '$'),
            ('hous_val_amt', 'Median house value (Census estimate for area)', '$'),

            # Education - Percentages
            ('per_less_than_9', '% of population with less than 9th grade education', '%'),
            ('per_9_to_12', '% with 9th-12th grade education (no high school diploma)', '%'),
            ('per_hsd', '% with high school diploma or equivalent (GED)', '%'),
            ('per_some_clg', '% with some college (no degree)', '%'),
            ('per_assct', '% with associate degree (2-year college)', '%'),
            ('per_bchlr', '% with bachelor degree (4-year college)', '%'),
            ('per_grad', '% with graduate or professional degree (Masters, PhD, etc.)', '%'),

            # Education - Counts
            ('edctn_less_than_9_qty', 'Count of people with <9th grade education', 'count'),
            ('edctn_9_12_qty', 'Count with 9th-12th grade (no diploma)', 'count'),
            ('edctn_high_schl_qty', 'Count with high school diploma', 'count'),
            ('edctn_some_clg_qty', 'Count with some college', 'count'),
            ('edctn_assct_dgre_qty', 'Count with associate degree', 'count'),
            ('edctn_bchlr_dgre_qty', 'Count with bachelor degree', 'count'),
            ('edctn_grad_dgre_qty', 'Count with graduate/professional degree', 'count'),

            # Occupation - Percentages
            ('per_mgmt', '% in management, business, science, arts occupations', '%'),
            ('per_svc', '% in service occupations (food, healthcare support, etc.)', '%'),
            ('per_sales', '% in sales and office occupations', '%'),
            ('per_prfsnl', '% in professional occupations (engineering, legal, medical, etc.)', '%'),
        ]

        for col, desc, unit in demo_features:
            if col in self.demographics.columns:
                self.analyze_numeric_feature(col, desc, unit, source="demographic")

    def analyze_correlations(self, top_n: int = 20):
        """Analyze correlations with target variable (price)"""
        self.print_header("CORRELATION WITH PRICE (All Features)")

        # Get numeric columns only
        numeric_cols = self.merged.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'price']

        # Calculate correlations
        correlations = {}
        for col in numeric_cols:
            corr = self.merged[['price', col]].corr().iloc[0, 1]
            correlations[col] = corr

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"\nTop {top_n} features by absolute correlation with price:\n")
        print(f"{'Rank':<6} {'Feature':<30} {'Correlation':<15} {'Strength'}")
        print("-" * 75)

        for idx, (col, corr) in enumerate(sorted_corr[:top_n], 1):
            strength = self._correlation_strength(abs(corr))
            direction = "↑ Positive" if corr > 0 else "↓ Negative"
            print(f"{idx:<6} {col:<30} {direction} {abs(corr):<7.3f}   {strength}")

        # Show weakest correlations too
        print(f"\n\nBottom 10 features by correlation (weakest predictors):\n")
        print(f"{'Rank':<6} {'Feature':<30} {'Correlation':<15} {'Strength'}")
        print("-" * 75)

        for idx, (col, corr) in enumerate(sorted_corr[-10:], 1):
            strength = self._correlation_strength(abs(corr))
            direction = "↑ Positive" if corr > 0 else "↓ Negative"
            print(f"{idx:<6} {col:<30} {direction} {abs(corr):<7.3f}   {strength}")

    def _correlation_strength(self, corr: float) -> str:
        """Classify correlation strength"""
        if corr >= 0.7:
            return "Very Strong ★★★★★"
        elif corr >= 0.5:
            return "Strong ★★★★"
        elif corr >= 0.3:
            return "Moderate ★★★"
        elif corr >= 0.1:
            return "Weak ★★"
        else:
            return "Very Weak ★"

    def analyze_data_quality(self):
        """Analyze data quality issues"""
        self.print_header("DATA QUALITY ANALYSIS")

        # Missing values
        print("Missing Values:")
        print("-" * 75)
        missing = self.merged.isna().sum()
        missing_pct = (missing / len(self.merged)) * 100

        has_missing = missing[missing > 0].sort_values(ascending=False)
        if len(has_missing) > 0:
            print(f"{'Column':<30} {'Missing Count':>15} {'Percentage':>12}")
            print("-" * 75)
            for col, count in has_missing.items():
                pct = missing_pct[col]
                print(f"{col:<30} {count:>15,} {pct:>11.1f}%")
        else:
            print("  ✓ No missing values found in any column!")

        # Duplicates
        print("\n\nDuplicate Records:")
        print("-" * 75)
        duplicates = self.sales.duplicated().sum()
        if duplicates > 0:
            print(f"  ⚠ Found {duplicates:,} duplicate records")
        else:
            print("  ✓ No duplicate records found")

        # Outliers and data anomalies
        print("\n\nPotential Outliers and Anomalies:")
        print("-" * 75)

        # Bedrooms
        max_bedrooms = self.sales['bedrooms'].max()
        if max_bedrooms > 10:
            outlier_count = (self.sales['bedrooms'] > 10).sum()
            print(f"  ⚠ Bedrooms: {outlier_count} houses with >10 bedrooms (max: {max_bedrooms:.0f})")
            print(f"     → Likely data error or commercial property")

        # Price extremes
        q99 = self.sales['price'].quantile(0.99)
        extreme_price = (self.sales['price'] > q99 * 1.5).sum()
        if extreme_price > 0:
            max_price = self.sales['price'].max()
            print(f"  ⚠ Price: {extreme_price} houses priced >1.5x 99th percentile")
            print(f"     → Max price: ${max_price:,.0f} (luxury/outlier properties)")

        # Sqft_living consistency check
        self.sales['sqft_calc'] = self.sales['sqft_above'] + self.sales['sqft_basement']
        mismatches = (abs(self.sales['sqft_living'] - self.sales['sqft_calc']) > 10).sum()
        if mismatches > 0:
            print(f"  ⚠ Square footage: {mismatches} records where sqft_living ≠ sqft_above + sqft_basement")
            print(f"     → Potential data entry errors")

        # Zero bathrooms
        zero_bath = (self.sales['bathrooms'] == 0).sum()
        if zero_bath > 0:
            print(f"  ⚠ Bathrooms: {zero_bath} houses with 0 bathrooms (likely error)")

        print("\n")

    def generate_summary(self):
        """Generate overall summary statistics"""
        self.print_header("DATASET SUMMARY")

        print("Dataset Overview:")
        print("-" * 75)
        print(f"  Total house sales: {len(self.sales):,}")
        print(f"  Date range: {self.sales['date'].min()} to {self.sales['date'].max()}")
        print(f"  Unique zipcodes: {self.sales['zipcode'].nunique()}")
        print(f"  Geographic coverage: King County, Washington")

        print(f"\nFeature Inventory:")
        print("-" * 75)
        print(f"  House features (numeric): 18")
        print(f"  House features (categorical): 2 (zipcode, date)")
        print(f"  Demographic features: 26")
        print(f"  Total features after merge: {len(self.merged.columns) - 1} (excluding zipcode)")
        print(f"  Target variable: price")

        print(f"\nPrice Distribution:")
        print("-" * 75)
        print(f"  Minimum: ${self.sales['price'].min():,.0f}")
        print(f"  25th percentile: ${self.sales['price'].quantile(0.25):,.0f}")
        print(f"  Median (50th): ${self.sales['price'].median():,.0f}")
        print(f"  Mean: ${self.sales['price'].mean():,.0f}")
        print(f"  75th percentile: ${self.sales['price'].quantile(0.75):,.0f}")
        print(f"  Maximum: ${self.sales['price'].max():,.0f}")
        print(f"  Std Dev: ${self.sales['price'].std():,.0f}")

        skew = self.sales['price'].mean() - self.sales['price'].median()
        print(f"\n  Skewness: Mean > Median by ${skew:,.0f} (right-skewed, luxury homes pull mean up)")

        print(f"\nKey Dataset Insights:")
        print("-" * 75)

        waterfront_pct = (self.sales['waterfront'] == 1).sum() / len(self.sales) * 100
        print(f"  • Waterfront: {waterfront_pct:.2f}% of properties (rare premium feature)")

        no_basement_pct = (self.sales['sqft_basement'] == 0).sum() / len(self.sales) * 100
        print(f"  • No basement: {no_basement_pct:.1f}% of houses have sqft_basement = 0")

        never_renovated_pct = (self.sales['yr_renovated'] == 0).sum() / len(self.sales) * 100
        print(f"  • Never renovated: {never_renovated_pct:.1f}% have yr_renovated = 0")

        no_view_pct = (self.sales['view'] == 0).sum() / len(self.sales) * 100
        print(f"  • No view: {no_view_pct:.1f}% have view rating = 0")

        median_age = 2025 - self.sales['yr_built'].median()
        print(f"  • Median house age: {median_age:.0f} years old (built in {self.sales['yr_built'].median():.0f})")

        print("\n" + "=" * 80 + "\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Comprehensive feature analysis for King County house sales data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--sales',
        type=str,
        default='data/kc_house_data.csv',
        help='Path to house sales CSV (default: data/kc_house_data.csv)'
    )
    parser.add_argument(
        '--demographics',
        type=str,
        default='data/zipcode_demographics.csv',
        help='Path to demographics CSV (default: data/zipcode_demographics.csv)'
    )
    parser.add_argument(
        '--top-corr',
        type=int,
        default=20,
        help='Number of top correlations to show (default: 20)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary and correlations (skip detailed feature analysis)'
    )

    args = parser.parse_args()

    # Initialize analyzer
    print("\nInitializing Feature Analyzer...")
    analyzer = FeatureAnalyzer(args.sales, args.demographics)

    # Run analyses
    analyzer.generate_summary()

    if not args.summary_only:
        analyzer.analyze_house_features()
        analyzer.analyze_demographic_features()

    analyzer.analyze_correlations(top_n=args.top_corr)
    analyzer.analyze_data_quality()

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
