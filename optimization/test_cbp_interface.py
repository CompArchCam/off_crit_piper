import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from cbp_interface import run_cbp, parse_cbp_output, calculate_cost

class TestCBPInterface(unittest.TestCase):
    
    # --- Parsing Tests ---
    def test_parse_output_valid(self):
        output = """
        Some unrelated text...
        FINAL_MISPREDICTIONS:
        cbp2016_tage_sc_l: 5678
        cbp2016_tage_no_sc: 339350
        cond_predictor_impl: 1234
        minitage: 321805
        tage_192: 296306
        """
        results = parse_cbp_output(output)
        self.assertEqual(results.get('cond_predictor_impl'), 1234)
        self.assertEqual(results.get('cbp2016_tage_sc_l'), 5678)
        self.assertEqual(results.get('minitage'), 321805)

    def test_parse_output_invalid(self):
        output = "No valid stats line here."
        with self.assertRaises(ValueError):
            parse_cbp_output(output)

    def test_calculate_cost_missing_keys(self):
        misp_dict = {'some_other_predictor': 100}
        with self.assertRaises(KeyError):
            calculate_cost(misp_dict)

    # --- Cost Calculation Tests ---
    def test_calculate_cost(self):
        # cost = cond_predictor_impl / cbp2016_tage_sc_l
        misp_dict = {
            'cond_predictor_impl': 100,
            'cbp2016_tage_sc_l': 200
        }
        cost = calculate_cost(misp_dict)
        self.assertEqual(cost, 0.5)

    def test_calculate_cost_zero_denominator(self):
        misp_dict = {
            'cond_predictor_impl': 100,
            'cbp2016_tage_sc_l': 0
        }
        cost = calculate_cost(misp_dict)
        self.assertEqual(cost, 1.0)

    # --- run_cbp Mocked Tests ---
    @patch('subprocess.run')
    def test_run_cbp_command_structure(self, mock_run):
        """
        Verify that run_cbp calls subprocess with the correct argument order:
        ../cbp <trace> -c <config>
        """
        mock_proc = MagicMock()
        mock_proc.stdout = "FINAL_MISPREDICTIONS:\ncond_predictor_impl: 10\ncbp2016_tage_sc_l: 20"
        mock_proc.stderr = ""
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        config_path = "test_config.txt"
        trace_path = "trce.gz"
        timeout = 123

        stdout, stderr, code = run_cbp(config_path, trace_path, timeout=timeout)

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        
        # Check command list
        cmd_list = args[0]
        self.assertEqual(cmd_list[0], '../cbp')
        self.assertEqual(cmd_list[1], trace_path)
        self.assertEqual(cmd_list[2], '-c')
        self.assertEqual(cmd_list[3], config_path)
        
        # Check timeout
        self.assertEqual(kwargs['timeout'], timeout)

    @patch('subprocess.run')
    def test_run_cbp_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='cmd', timeout=10)
        
        with self.assertRaises(TimeoutError):
            run_cbp('cfg', 'trace', timeout=10)

    # --- Integration Test (Real execution) ---
    def test_integration_run(self):
        """
        Tries to run the actual CBP binary if it exists.
        Skipped if binary or traces are missing.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cbp_path = os.path.join(base_dir, 'cbp')
        config_path = os.path.join(base_dir, 'example.config')
        # One valid trace path
        trace_path = os.path.join(base_dir, 'sample_traces/fp/sample_fp_trace.gz')

        if not os.path.exists(cbp_path):
            self.skipTest(f"CBP binary not found at {cbp_path}")
        if not os.path.exists(config_path):
            self.skipTest(f"Config not found at {config_path}")
        if not os.path.exists(trace_path):
            self.skipTest(f"Trace not found at {trace_path}")

        # Run with a short timeout, just to see if it starts and outputs something
        # Note: 1s might be too short for full run, but enough to catch startup errors.
        # But run_cbp waits for completion. So we'll run it, assuming it finishes quickly or we accept it takes time.
        # The sample trace is smallish but might take >1s. Let's give it 10s.
        try:
            # We need to call run_cbp from the 'optimization' dir context or adjust paths in run_cbp?
            # run_cbp hardcodes '../cbp'. So we must run this test from 'optimization/' dir
            # or Ensure run_cbp logic holds.
            # run_cbp uses relative path '../cbp'. 
            # If we run this test from 'optimization' directory, it works.
            
            # Let's adjust CWD for the duration of this test if needed, 
            # but usually tests are run from root or we adjust imports. 
            # Assuming we run `python3 optimization/test_cbp_interface.py` from root? 
            # Or `cd optimization && python3 test_cbp_interface.py`.
            # The tool usually runs commands in a specific CWD.
            
            # We'll assume the standard usage: running from optimization dir.
            # But let's verify where we are.
            cwd = os.getcwd()
            if os.path.basename(cwd) != 'optimization':
                # If we are in root, run_cbp's '../cbp' will look in parent of root -> fail.
                # So we should skip or warn.
                # But typically we run tests where the code is.
                pass 

            stdout, stderr, returncode = run_cbp(config_path, trace_path, timeout=30)
            
            if returncode != 0:
                self.fail(f"Integration run failed with code {returncode}. Stderr: {stderr}")
            
            results = parse_cbp_output(stdout)
            self.assertIn('cond_predictor_impl', results)
            self.assertIn('cbp2016_tage_sc_l', results)
            
        except TimeoutError:
             self.fail("Integration run timed out (30s)")
        except Exception as e:
            self.fail(f"Integration run crashed: {e}")

if __name__ == '__main__':
    unittest.main()
