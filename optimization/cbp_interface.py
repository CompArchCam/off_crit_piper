import subprocess
import re
import os

def run_cbp(config_path, trace_path, timeout=600):
    """
    Runs the CBP executable with the given config and trace.
    Returns the process output (stdout) or raises an exception on failure/timeout.
    """
    cmd = ['../cbp', trace_path, '-c', config_path]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"CBP execution timed out after {timeout}s")

def parse_cbp_output(stdout):
    """
    Parses the stdout from CBP to extract misprediction counts.
    Expected format is the FINAL_MISPREDICTIONS section.
    Returns: dictionary mapping predictor names to their misprediction counts.
    """
    results = {}
    if "FINAL_MISPREDICTIONS:" in stdout:
        section = stdout.split("FINAL_MISPREDICTIONS:")[1]
        lines = section.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, val = line.split(":")
                try:
                    results[key.strip()] = int(val.strip())
                except ValueError:
                    continue
    
    if not results:
        raise ValueError("Could not find FINAL_MISPREDICTIONS section in CBP output")
            
    return results

def calculate_cost(misp_dict):
    """
    Calculates the cost function for optimization.
    Formula: cond_predictor_impl / cbp2016_tage_sc_l
    """
    my_misp = misp_dict['cond_predictor_impl']
    ref_misp = misp_dict['cbp2016_tage_sc_l']
        
    if ref_misp == 0:
        return 1.0 # Avoid division by zero
        
    return my_misp / ref_misp
