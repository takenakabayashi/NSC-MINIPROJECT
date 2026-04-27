import matplotlib.pyplot as plt

results = {
    "Naive": 7.1746,
    "Numpy": 1.7494,
    "Numba f32": 0.0957,
    "Numba f64": 0.0880,
    "Parallel": 0.0294,
    "Dask Local": 0.0694,
    "Dask Cluster": 0.1611,
    "GPU f32": 0.0032,
    "GPU f64": 0.0074
}

names,times = zip(*results.items())
plt.bar(names, times, log=True)
plt.ylabel("seconds (log scale)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()