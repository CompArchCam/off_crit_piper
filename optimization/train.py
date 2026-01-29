#!/usr/bin/env python3
import tempfile
import os
import time
import random
import resource
import logging
from smac import HyperparameterOptimizationFacade, Scenario
from pspace import parse_pspace, config_to_file
from cbp_interface import run_cbp, parse_cbp_output, calculate_cost

# Configure logging to only show warnings and errors
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PSPACE_FILE = 'config.pspace'
OUTPUT_FILE = 'optimized_config.txt'
# Using sample traces available in the repo
TRACES = [
    '../sample_traces/fp/sample_fp_trace.gz',
    '../sample_traces/int/sample_int_trace.gz'
]
TRACE_PER_RUN = 2 # Use both for stability in this small set
N_TRIALS = 50 # Short run for demonstration/testing
N_WORKERS = 4 

# Increase file limit if possible
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
except Exception:
    pass

# Parse the pspace template
try:
    template, config_space = parse_pspace(PSPACE_FILE)
except Exception as e:
    logger.error(f"Failed to parse pspace file: {e}")
    exit(1)

def evaluate(config, seed: int = 0) -> float:
    """
    Evaluates a configuration by running CBP on sample traces.
    Returns the average cost (combined_misp / tage_misp).
    """
    # Create temporary config file
    fd, config_path = tempfile.mkstemp(suffix='.txt', text=True)
    os.close(fd)
    
    try:
        # Write config to file
        config_to_file(template, config, config_path)

        costs = []
        # Run on traces
        sampled_traces = TRACES # For now use all (2) traces since set is small
        
        for trace_path in sampled_traces:
            try:
                stdout, stderr, returncode = run_cbp(config_path, trace_path, timeout=600)
                
                if returncode != 0:
                    logger.error(f"CBP failed on {trace_path}. Code: {returncode}")
                    # logger.debug(stderr) 
                    return 1e9 # High cost for failure

                combined, tage = parse_cbp_output(stdout)
                
                if combined is None or tage is None:
                    logger.error(f"Failed to parse output on {trace_path}")
                    return 1e9
                
                cost = calculate_cost(combined, tage)
                costs.append(cost)

            except Exception as e:
                logger.error(f"Exception running trace {trace_path}: {e}")
                return 1e9
            
        if not costs:
            return 1e9
            
        return sum(costs) / len(costs)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1e9
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == '__main__':
    # Removed informational prints to reduce contention
    
    scenario = Scenario(
        configspace=config_space,
        deterministic=False,
        n_trials=N_TRIALS,
        n_workers=N_WORKERS,
        name="cbp_optimization"
    )

    smac = HyperparameterOptimizationFacade(scenario=scenario, target_function=evaluate)
    incumbent = smac.optimize()

    # Save Best Result
    final_cost = evaluate(incumbent)
    config_to_file(template, incumbent, OUTPUT_FILE)
