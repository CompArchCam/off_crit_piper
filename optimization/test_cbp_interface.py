import unittest
from cbp_interface import parse_cbp_output, calculate_cost

class TestCBPInterface(unittest.TestCase):
    def test_parse_output_valid(self):
        output = """
        Some unrelated text...
        ReferenceMispred/Mispred: 1234/5678
        More text...
        """
        combined, tage = parse_cbp_output(output)
        self.assertEqual(combined, 1234)
        self.assertEqual(tage, 5678)

    def test_parse_output_invalid(self):
        output = "No valid stats line here."
        combined, tage = parse_cbp_output(output)
        self.assertIsNone(combined)
        self.assertIsNone(tage)

    def test_calculate_cost(self):
        cost = calculate_cost(100, 200)
        self.assertEqual(cost, 0.5)

    def test_calculate_cost_zero_ref(self):
        cost = calculate_cost(100, 0)
        self.assertEqual(cost, 1.0)

if __name__ == '__main__':
    unittest.main()
