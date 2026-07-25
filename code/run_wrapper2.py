"""Wrapper to run experiments with explicit flushing."""
import sys
import os
import io

# Make stdout unbuffered
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', write_through=True, line_buffering=True)

# Set argv
sys.argv = ['run_fixed_experiments.py', '--seed', '42', '--split', 'temporal']

# Now import and run
print("Starting experiment...", flush=True)
import run_fixed_experiments
