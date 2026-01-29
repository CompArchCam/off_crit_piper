import unittest
import sys
import os

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import evaluate, config_space

class TestTrainEvaluate(unittest.TestCase):
    def test_evaluate_default_config(self):
        """
        Tests the evaluate function using the default configuration from the config space.
        This runs the full CBP pipeline on the sample traces.
        """
        print("\nSampling default configuration from config_space...")
        config = config_space.get_default_configuration()
        print(f"Configuration: {config}")

        print("\nRunning evaluate()...")
        # seed is optional but good practice to provide
        cost = evaluate(config, seed=42)
        
        print(f"\nResulting Cost: {cost}")
        
        # Basic assertions
        self.assertIsInstance(cost, float)
        self.assertLess(cost, 1e9, "Evaluation returned failure cost (1e9). Check if CBP binary works and traces exist.")

if __name__ == '__main__':
    unittest.main()
