from benchmark import benchmark

import time
import numpy as np

def compute_rows(M):
    for i in range(N):
        s = np.sum(M[i, :])

def compute_columns(M):
    for i in range(N):
        s = np.sum(M[:, i])
        
if __name__ == "__main__":
    time_start = time.time()
        
    N = 10000
    A = np.random.rand(N, N)
    
    med_time, res = benchmark(compute_rows, A, n_runs=1)
    print(f"Execution time compute_rows(): {med_time:.2f} seconds")
    
    med_time, res = benchmark(compute_columns, A, n_runs=1)
    print(f"Execution time compute_columns(): {med_time:.2f} seconds")
    
    med_time, res = benchmark(np.asfortranarray, A, n_runs=1)
    print(f"Execution time np.asfortranarray(): {med_time:.2f} seconds")
    
    
"""
compute_rows() runs slightly faster ~0.19 seconds, due to it running row-major, and so it performs stride 1 where it accesses the 
entire cache line instead of only a couple

compute_columns() runs slower ~1.46 seconds, due to it running column-major, and so it performs stride 3 where it accesses only
a couple elements at a time in the cache line

np.asfortranarray() also runs pretty slow ~1.5 seconds, presumably due to it also being column-major
"""