"""Wrapper to run experiments with output redirected to file."""
import sys
import os
import contextlib

log_path = os.path.join(os.path.dirname(__file__), 'run_seed42_direct.log')
sys.argv = ['run_fixed_experiments.py', '--seed', '42', '--split', 'temporal']

with open(log_path, 'w', encoding='utf-8') as f:
    # Redirect both stdout and stderr
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            exec(open('run_fixed_experiments.py').read())
        except Exception as e:
            f.write(f"\n\nERROR: {type(e).__name__}: {e}\n")
            import traceback
            f.write(traceback.format_exc())
            f.flush()

print(f"Done. Log saved to {log_path}")
