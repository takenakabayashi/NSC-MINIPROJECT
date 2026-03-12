import time
import statistics
import os
import sys
import yaml
import numpy as np
from mandelbrot_parallel import mandelbrot_serial, mandelbrot_parallel

def compute_lif(p, tp, t1):
    """
    Compute Load Imbalance Factor
    
    LIF = p · (T_p / T_1) - 1
    
    where:
      p = number of workers
      T_p = time with p workers
      T_1 = serial time
    """
    return p * (tp / t1) - 1

def run_serial_baseline(x_min, x_max, y_min, y_max, N, max_iter, n_runs=3):
    """
    Measure serial baseline time (T_1)
    """
    print(f"\nMeasuring SERIAL baseline (grid size: {N}x{N})...")
    times = []
    
    for i in range(n_runs):
        t0 = time.perf_counter()
        _ = mandelbrot_serial(x_min, x_max, y_min, y_max, N, max_iter)
        times.append(time.perf_counter() - t0)
    
    t_serial = statistics.median(times)
    print(f"  Serial median time: {t_serial:.4f}s (min={min(times):.4f}, max={max(times):.4f})")
    return t_serial

def run_parallel_benchmark(x_min, x_max, y_min, y_max, N, max_iter, 
                          num_workers, n_chunks, n_runs=3, warmup=True):
    """
    Benchmark parallel implementation with given configuration
    """
    if warmup:
        # Warm-up run (not counted)
        _ = mandelbrot_parallel(x_min, x_max, y_min, y_max, N, max_iter, 
                               num_workers, n_chunks)
    
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        _ = mandelbrot_parallel(x_min, x_max, y_min, y_max, N, max_iter, 
                               num_workers, n_chunks)
        times.append(time.perf_counter() - t0)
    
    t_parallel = statistics.median(times)
    return t_parallel

def benchmark_chunk_sweep(x_min, x_max, y_min, y_max, N, max_iter, 
                         best_workers, n_runs=3):
    """
    Sweep through n_chunks: 1×, 2×, 4×, 8×, 16× workers
    """
    print(f"\n{'n chunks':>10} {'n_chunks val':>14} {'Time (s)':>12} {'vs. 1x LIF':>12} {'LIF':>10}")
    print("-" * 60)
    
    # Get serial baseline
    t_serial = run_serial_baseline(x_min, x_max, y_min, y_max, N, max_iter, n_runs)
    
    results = []
    multipliers = [1, 2, 4, 8, 16]
    baseline_time = None
    
    for mult in multipliers:
        n_chunks = mult * best_workers
        
        print(f"\nBenchmarking with n_chunks={n_chunks} ({mult}× workers)...")
        t_parallel = run_parallel_benchmark(x_min, x_max, y_min, y_max, N, max_iter,
                                           best_workers, n_chunks, n_runs)
        
        # Compute metrics
        speedup = t_serial / t_parallel
        lif = compute_lif(best_workers, t_parallel, t_serial)
        efficiency = (speedup / best_workers) * 100
        
        # For comparison with 1× baseline
        if baseline_time is None:
            baseline_time = t_parallel
            time_vs_baseline = 1.0
        else:
            time_vs_baseline = t_parallel / baseline_time
        
        results.append({
            'multiplier': mult,
            'n_chunks': n_chunks,
            'time': t_parallel,
            'speedup': speedup,
            'efficiency': efficiency,
            'lif': lif,
            'time_vs_baseline': time_vs_baseline
        })
        
        print(f"  {mult:>2}× {n_chunks:>6d}       {t_parallel:>10.4f}s  "
              f"{time_vs_baseline:>10.2f}x    {lif:>8.4f}")
    
    return t_serial, results

def print_summary(t_serial, results, best_workers):
    """
    Print comprehensive summary table
    """
    print("\n" + "="*75)
    print("SUMMARY: LIF Analysis for Load Imbalance")
    print("="*75)
    print(f"Fixed workers: {best_workers}")
    print(f"Serial baseline time (T_1): {t_serial:.4f}s\n")
    
    print(f"{'n chunks':>10} {'Value':>8} {'Time (s)':>12} {'Speedup':>10} {'Eff. %':>8} {'LIF':>10}")
    print("-" * 75)
    
    for r in results:
        print(f"{r['multiplier']:>2}× {r['n_chunks']:>6d}x     {r['time']:>10.4f}s  "
              f"{r['speedup']:>8.2f}x    {r['efficiency']:>6.1f}%   {r['lif']:>8.4f}")
    
    # Find optimal (minimum LIF)
    min_lif_result = min(results, key=lambda x: x['lif'])
    print("\n" + "-"*75)
    print(f"✓ OPTIMAL: {min_lif_result['multiplier']}× workers ({min_lif_result['n_chunks']} chunks)")
    print(f"  LIF = {min_lif_result['lif']:.4f} (minimum = best balance)")
    print(f"  Time: {min_lif_result['time']:.4f}s")
    print(f"  Efficiency: {min_lif_result['efficiency']:.1f}%")
    print("="*75)
    
    return min_lif_result

if __name__ == "__main__":
    # Load region configuration
    with open("mandelbrot_regions.yml", 'r') as f:
        yaml_file = yaml.full_load(f)
    
    regions_dict = yaml_file.get("Regions", {})
    
    # Use default region or prompt
    print("Available regions:", list(regions_dict.keys()))
    region_name = input("Enter region name (default: 'Full'): ").strip() or "Full"
    region_name = region_name.title()
    
    if region_name not in regions_dict:
        print(f"Region '{region_name}' not found. Using 'Full'")
        region_name = "Full"
    
    region = regions_dict[region_name]
    
    # Parameters
    x_min, x_max = region['x_min'], region['x_max']
    y_min, y_max = region['y_min'], region['y_max']
    max_iter = region['max_iter']
    
    # Grid size for benchmarking (use a reasonable size)
    grid_size = int(input("Enter grid size (default 1024): ") or 1024)
    
    # Get best number of workers
    best_workers = 4
    
    # Run benchmark sweep
    t_serial, results = benchmark_chunk_sweep(
        x_min, x_max, y_min, y_max, grid_size, max_iter, 
        best_workers, n_runs=3
    )
    
    # Print and save results
    optimal = print_summary(t_serial, results, best_workers)
    
    print("\n✓ Done! Record optimal n_chunks and LIF in your performance notebook.")
