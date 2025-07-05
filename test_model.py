import unittest
import numpy as np
from improved_model import create_improved_model, add_engineered_features, main
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
import pandas as pd
import joblib
import os

class TestModelPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run the model training once before all tests"""
        # Run main to train and save the model
        main()
        
        # Load the trained model and preprocessor
        cls.model = tf.keras.models.load_model('improved_model.keras')
        cls.preprocessor = joblib.load('preprocessor.joblib')
        
        # Load and prepare test data
        df = pd.read_csv('final_data.csv')
        numeric_features = ['cost_of_living', 'infrastructure_level', 'happiness_level']
        categorical_features = ['experience_level']
        
        X = df[numeric_features + categorical_features]
        cls.y = df['mean_salary']
        
        # Preprocess the data
        X_processed = cls.preprocessor.transform(X)
        cls.X_final = add_engineered_features(X_processed)
        
        # Get predictions
        cls.y_pred = cls.model.predict(cls.X_final)

    def test_mse_threshold(self):
        """Test if MSE is below acceptable threshold"""
        mse = mean_squared_error(self.y, self.y_pred)
        print(f"\nActual MSE: {mse:.2f}")
        self.assertLess(mse, 5000000)  # Adjust threshold based on expected performance

    def test_r2_score_threshold(self):
        """Test if R² score is above acceptable threshold"""
        r2 = r2_score(self.y, self.y_pred)
        print(f"Actual R²: {r2:.4f}")
        self.assertGreater(r2, 0.5)  # Model should explain at least 50% of variance

    def test_mae_threshold(self):
        """Test if MAE is below acceptable threshold"""
        mae = mean_absolute_error(self.y, self.y_pred)
        print(f"Actual MAE: {mae:.2f}")
        self.assertLess(mae, 50000)  # Average prediction should be within $50,000

    def test_prediction_range(self):
        """Test if predictions are within reasonable salary range"""
        min_pred = np.min(self.y_pred)
        max_pred = np.max(self.y_pred)
        print(f"Prediction range: ${min_pred:.2f} - ${max_pred:.2f}")
        self.assertGreater(min_pred, 0)  # Salaries should be positive
        self.assertLess(max_pred, 1000000)  # Assuming no salaries over $1M in dataset

    def test_model_architecture(self):
        """Test if model architecture matches expected structure"""
        model = create_improved_model(self.X_final.shape[1])
        self.assertEqual(len(model.layers), 7)  # Should have 7 layers
        self.assertEqual(model.count_params(), self.model.count_params())

    @classmethod
    def tearDownClass(cls):
        """Clean up saved model files after tests"""
        try:
            os.remove('improved_model.keras')
            os.remove('preprocessor.joblib')
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main(verbosity=2) 