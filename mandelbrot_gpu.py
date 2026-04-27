import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time

def mandelbrot_gpu_float32(x_min, x_max, y_min, y_max, N, max_iter):
    KERNEL_SRC = """
    __kernel void mandelbrot(
        __global int *result,
        const float x_min, const float x_max,
        const float y_min, const float y_max,
        const int N, const int max_iter) 
    {
        int col = get_global_id(0);
        int row = get_global_id(1);
        if(col >= N || row >= N) return;
        
        float c_real = x_min + col * (x_max - x_min) / (float) N;
        float c_imag = y_min + row * (y_max - y_min) / (float) N;
        
        float zr = 0.0f, zi = 0.0f;
        int count = 0;
        
        while(count < max_iter && zr*zr + zi*zi <= 4.0f) {
            float tmp = zr*zr - zi*zi + c_real;
            zi = 2.0f * zr * zi + c_imag;
            zr = tmp;
            count++;
        }
        
        result[row * N + col] = count;
    }
    """
    
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, KERNEL_SRC).build()
    
    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)
    
    # Warm up to first trigger the kernel compile
    prog.mandelbrot(
        queue, (N, N), None,
        image_dev,
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(N), np.int32(max_iter)
    )
    queue.finish()
    
    # Time measurement
    t_start = time.perf_counter()
    prog.mandelbrot(
        queue, (N, N), None,
        image_dev,
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(N), np.int32(max_iter)
    )
    queue.finish()
    t_end = time.perf_counter() - t_start
    
    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()
    
    return image, t_end

def mandelbrot_gpu_float64(x_min, x_max, y_min, y_max, N, max_iter):
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    KERNEL_SRC = """
    __kernel void mandelbrot(
        __global int *result,
        const double x_min, const double x_max,
        const double y_min, const double y_max,
        const int N, const int max_iter) 
    {
        int col = get_global_id(0);
        int row = get_global_id(1);
        if(col >= N || row >= N) return;
        
        double c_real = x_min + col * (x_max - x_min) / (double) N;
        double c_imag = y_min + row * (y_max - y_min) / (double) N;
        
        double zr = 0.0, zi = 0.0;
        int count = 0;
        
        while(count < max_iter && zr*zr + zi*zi <= 4.0) {
            double tmp = zr*zr - zi*zi + c_real;
            zi = 2.0 * zr * zi + c_imag;
            zr = tmp;
            count++;
        }
        
        result[row * N + col] = count;
    }
    """
    
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, KERNEL_SRC).build()
    
    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)
    
    # Warm up to first trigger the kernel compile
    prog.mandelbrot(
        queue, (N, N), None,
        image_dev,
        np.float64(x_min), np.float64(x_max),
        np.float64(y_min), np.float64(y_max),
        np.int32(N), np.int32(max_iter)
    )
    queue.finish()
    
    # Time measurement
    t_start = time.perf_counter()
    prog.mandelbrot(
        queue, (N, N), None,
        image_dev,
        np.float64(x_min), np.float64(x_max),
        np.float64(y_min), np.float64(y_max),
        np.int32(N), np.int32(max_iter)
    )
    queue.finish()
    t_end = time.perf_counter() - t_start
    
    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()
    
    return image, t_end

def visualize_mandelbrot(mandelbrot_set, title, x_min, x_max, y_min, y_max, colormap):
    """
    Function to visualize Mandelbrot Set with a colormap
    
    :param mandelbrot_set: 2D array of Mandelbrot Set results
    :param x_min: Minimum x value of the complex plane
    :param x_max: Maximum x value of the complex plane
    :param y_min: Minimum y value of the complex plane
    :param y_max: Maximum y value of the complex plane
    :param colormap: Colormap to use for visualization
    """
    plt.imshow(mandelbrot_set, extent=[x_min, x_max, y_min, y_max], cmap=colormap, origin='lower', aspect='equal')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    
if __name__ == "__main__":
    # Load available regions from YAML file
    with open("mandelbrot_regions.yml", 'r') as f:
        yaml_file = yaml.full_load(f)
    
    regions_dict = yaml_file.get("Regions", {})
    
    # Display available regions and prompt user to select one
    print("Regions: ", list(regions_dict.keys()))
    region = input("Choose a region: ").lower().title()
    
    if(len(regions_dict[region]) > 0):
        region = regions_dict[region]
    
    # initialize parameters
    x_min, x_max = region['x_min'], region['x_max']
    y_min, y_max = region['y_min'], region['y_max']
    max_iter = region['max_iter']
    grid_sizes = ['256', '512', '1024', '2048', '4096', '8192']
    implementations = ['float32', 'float64']
    execution_times = []
    show_plots = False
    
    for impl in implementations:
        if impl == 'float32':
            mandelbrot_func = mandelbrot_gpu_float32
        elif impl == 'float64':
            mandelbrot_func = mandelbrot_gpu_float64
            
        for gr_size in grid_sizes:
            width, height = int(gr_size), int(gr_size)
            print(f"Computing Mandelbrot set for grid size: {gr_size}x{gr_size} with {impl} precision...")
            mandelbrot_set, exec_time = mandelbrot_func(x_min, x_max, y_min, y_max, width, max_iter)
            execution_times.append((impl, gr_size, exec_time))
            print(f"Execution time: {exec_time:.4f} seconds")
            
            if show_plots:
                title = f"Mandelbrot Set ({impl}, {gr_size}x{gr_size})"
                visualize_mandelbrot(mandelbrot_set, title, x_min, x_max, y_min, y_max, colormap='inferno')
        