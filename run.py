import subprocess
import sys

# run the test_on_clusters script 10 times, so we can eventually average across 10 runs
script_name = 'test_on_clusters.py'
n_iter = 6

for i in range(n_iter):
    subprocess.call(['python', script_name], stdout=sys.stdout, stderr=subprocess.STDOUT)