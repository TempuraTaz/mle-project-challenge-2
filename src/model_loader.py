"""
Model Loader Module

Handles loading and preparation of ML model artifacts for the house price prediction API.
This module encapsulates all model loading logic, ensuring consistency and proper error handling.

Components loaded:
- Trained ML model (scikit-learn pipeline)
- Model features (expected input order)
- Feature defaults (for /predict-minimal endpoint)
- Demographics data (for zipcode-based feature engineering)

Usage:
    from src.model_loader import ModelLoader

    loader = ModelLoader()
    artifacts = loader.load_all()

    # Access loaded components
    prediction = artifacts.model.predict(X)
    features = artifacts.model_features
    defaults = artifacts.feature_defaults
    demographics = artifacts.demographics
"""

import json
import pickle
import logging
from dataclasses import dataclass
from typing import Any, Dict, List
import pandas as pd

# Import centralized config
from src.config import Config

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Container for all loaded model artifacts"""
    model: Any  # Scikit-learn model/pipeline
    model_features: List[str]  # Feature names in correct order
    feature_defaults: Dict[str, float]  # Default values for missing features
    demographics: pd.DataFrame  # Demographics data for zipcode join
    model_version: str  # Model version identifier


class ModelLoader:
    """
    Loads and validates all model artifacts required for predictions.

    This class handles:
    - Loading the trained ML model
    - Loading model metadata (features, defaults)
    - Loading supporting data (demographics)
    - Validation of all artifacts
    - Proper error handling with informative messages
    """

    def __init__(self):
        """Initialize the model loader with paths from config"""
        self.model_path = Config.MODEL_PATH
        self.features_path = Config.MODEL_FEATURES_PATH
        self.defaults_path = Config.DEFAULTS_PATH
        self.demographics_path = Config.DEMOGRAPHICS_PATH
        self.model_version = Config.MODEL_VERSION

    def load_model(self) -> Any:
        """
        Load the trained scikit-learn model from pickle file.

        Returns:
            Trained model/pipeline

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading model from {self.model_path}")

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. "
                    "Please run 'python scripts/create_model.py' to train the model."
                )

            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            logger.info("✓ Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_features(self) -> List[str]:
        """
        Load the list of model features in the correct order.

        Returns:
            List of feature names

        Raises:
            FileNotFoundError: If features file doesn't exist
            Exception: If features loading fails
        """
        try:
            logger.info(f"Loading model features from {self.features_path}")

            if not self.features_path.exists():
                raise FileNotFoundError(
                    f"Features file not found at {self.features_path}. "
                    "Please run 'python scripts/create_model.py' to generate it."
                )

            with open(self.features_path, "r") as f:
                features = json.load(f)

            logger.info(f"✓ Loaded {len(features)} features")
            return features

        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            raise

    def load_defaults(self) -> Dict[str, float]:
        """
        Load feature defaults for handling missing values.

        Returns:
            Dictionary mapping feature names to default values

        Raises:
            FileNotFoundError: If defaults file doesn't exist
            Exception: If defaults loading fails
        """
        try:
            logger.info(f"Loading feature defaults from {self.defaults_path}")

            if not self.defaults_path.exists():
                raise FileNotFoundError(
                    f"Defaults file not found at {self.defaults_path}. "
                    "Please run 'python scripts/create_model.py' to generate it."
                )

            with open(self.defaults_path, "r") as f:
                defaults = json.load(f)

            logger.info(f"✓ Loaded defaults for {len(defaults)} features")
            return defaults

        except Exception as e:
            logger.error(f"Failed to load defaults: {e}")
            raise

    def load_demographics(self) -> pd.DataFrame:
        """
        Load demographics data for zipcode-based feature engineering.

        Returns:
            DataFrame with demographics indexed by zipcode

        Raises:
            FileNotFoundError: If demographics file doesn't exist
            Exception: If demographics loading fails
        """
        try:
            logger.info(f"Loading demographics from {self.demographics_path}")

            if not self.demographics_path.exists():
                raise FileNotFoundError(
                    f"Demographics file not found at {self.demographics_path}. "
                    "Please ensure data files are in the data/ directory."
                )

            demographics = pd.read_csv(
                self.demographics_path,
                dtype={"zipcode": str}
            )

            logger.info(f"✓ Loaded demographics for {len(demographics)} zipcodes")
            return demographics

        except Exception as e:
            logger.error(f"Failed to load demographics: {e}")
            raise

    def load_all(self) -> ModelArtifacts:
        """
        Load all model artifacts in one operation.

        This is the main entry point for loading everything needed
        for the prediction API.

        Returns:
            ModelArtifacts dataclass containing all loaded components

        Raises:
            Exception: If any artifact fails to load
        """
        try:
            logger.info("=" * 60)
            logger.info("LOADING MODEL ARTIFACTS")
            logger.info("=" * 60)

            # Load all components
            model = self.load_model()
            features = self.load_features()
            defaults = self.load_defaults()
            demographics = self.load_demographics()

            # Package everything together
            artifacts = ModelArtifacts(
                model=model,
                model_features=features,
                feature_defaults=defaults,
                demographics=demographics,
                model_version=self.model_version
            )

            logger.info("=" * 60)
            logger.info("✓ All artifacts loaded successfully")
            logger.info(f"Model version: {self.model_version}")
            logger.info(f"Features: {len(features)}")
            logger.info(f"Zipcodes: {len(demographics)}")
            logger.info("=" * 60)

            return artifacts

        except Exception as e:
            logger.error("=" * 60)
            logger.error("✗ FAILED TO LOAD MODEL ARTIFACTS")
            logger.error(f"Error: {e}")
            logger.error("=" * 60)
            raise


# Convenience function for quick loading
def load_model_artifacts() -> ModelArtifacts:
    """
    Convenience function to load all model artifacts.

    Returns:
        ModelArtifacts dataclass containing all components
    """
    loader = ModelLoader()
    return loader.load_all()


if __name__ == "__main__":
    # Test the loader when run directly
    logging.basicConfig(level=logging.INFO)
    print("\nTesting Model Loader...")
    print("=" * 60)

    try:
        artifacts = load_model_artifacts()
        print("\n✓ Model loader test successful!")
        print(f"Model type: {type(artifacts.model)}")
        print(f"Number of features: {len(artifacts.model_features)}")
        print(f"Number of zipcodes: {len(artifacts.demographics)}")
    except Exception as e:
        print(f"\n✗ Model loader test failed: {e}")
