#!/usr/bin/env python3
import tempfile
import os
import time
import random
import resource
from smac import HyperparameterOptimizationFacade, Scenario
from pspace import parse_pspace, config_to_file
from cbp_interface import run_cbp, parse_cbp_output, calculate_cost

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
    print(f"Failed to parse pspace file: {e}")
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
                stdout, stderr, returncode = run_cbp(config_path, trace_path, timeout=60)
                
                if returncode != 0:
                    print(f"CBP failed on {trace_path}. Code: {returncode}")
                    # print(stderr) 
                    return 1e9 # High cost for failure

                combined, tage = parse_cbp_output(stdout)
                
                if combined is None or tage is None:
                    print(f"Failed to parse output on {trace_path}")
                    return 1e9
                
                cost = calculate_cost(combined, tage)
                costs.append(cost)

            except Exception as e:
                print(f"Exception running trace {trace_path}: {e}")
                return 1e9
            
        if not costs:
            return 1e9
            
        return sum(costs) / len(costs)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1e9
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == '__main__':
    print(f"Starting optimization with {N_WORKERS} workers...")
    print(f"Traces: {TRACES}")
    
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
    print("\noptimization finished.")
    final_cost = evaluate(incumbent)
    print(f"Best cost: {final_cost:.4f}")
    config_to_file(template, incumbent, OUTPUT_FILE)
    print(f"Saved optimized config to: {OUTPUT_FILE}")
