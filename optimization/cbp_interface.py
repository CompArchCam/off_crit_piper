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
    Expected format line: "ReferenceMispred/Mispred: <combined>/<tage>"
    Returns: (combined_mispredictions, tage_mispredictions)
    """
    match = re.search(r'ReferenceMispred/Mispred:\s+(\d+)/(\d+)', stdout)
    if match:
        combined_misp = int(match.group(1))
        tage_misp = int(match.group(2))
        return combined_misp, tage_misp
    return None, None

def calculate_cost(combined_misp, tage_misp):
    """
    Calculates the cost function for optimization.
    Returns: combined_misp / tage_misp
    """
    if tage_misp == 0:
        return 1.0 # Should not happen usually, but avoid div by zero
    return combined_misp / tage_misp
