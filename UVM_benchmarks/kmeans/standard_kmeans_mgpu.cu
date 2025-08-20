#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string>

double std_time_used;
struct Data {
  Data() {}

  Data(int size) : size(size), bytes(size * sizeof(double)) {
    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);
    cudaMemset(x, 0, bytes);
    cudaMemset(y, 0, bytes);
  }

  Data(int size, std::vector<double>& h_x, std::vector<double>& h_y)
  : size(size), bytes(size * sizeof(double)) {
    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);
    // cudaMemcpy(x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(y, h_y.data(), bytes, cudaMemcpyHostToDevice);
    memcpy(x, h_x.data(), bytes);
    memcpy(y, h_y.data(), bytes);
  }

  ~Data() {
    cudaFree(x);
    cudaFree(y);
  }

  double* x{nullptr};
  double* y{nullptr};
  int size{0};
  int bytes{0};
};

__device__ double
squared_l2_distance(double x_1, double y_1, double x_2, double y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__global__ void fine_reduce(const double* __restrict__ data_x,
                            const double* __restrict__ data_y,
                            int data_size,
                            const double* __restrict__ means_x,
                            const double* __restrict__ means_y,
                            double* __restrict__ new_sums_x,
                            double* __restrict__ new_sums_y,
                            int k,
                            int* __restrict__ counts) {
  extern __shared__ double shared_data[];

  const int local_index = threadIdx.x; 
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= data_size) return;

  // Load the centroids into shared memory.
  if (local_index < k) {
    shared_data[local_index] = means_x[local_index];      // x pos of centroid
    shared_data[k + local_index] = means_y[local_index];  // y pos of centroid
  }

  __syncthreads();

  // Load once here.
  const double x_value = data_x[global_index];
  const double y_value = data_y[global_index];

  // each thread deals with only one point
  // each point get their best centroid
  double best_distance = FLT_MAX;
  int best_cluster = -1;
  for (int cluster = 0; cluster < k; ++cluster) {
    const double distance = squared_l2_distance(x_value,
                                               y_value,
                                               shared_data[cluster],
                                               shared_data[k + cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  __syncthreads();

  // reduction

  const int x = local_index;
  const int y = local_index + blockDim.x;
  const int count = local_index + blockDim.x + blockDim.x;

  for (int cluster = 0; cluster < k; ++cluster) {
    shared_data[x] = (best_cluster == cluster) ? x_value : 0;
    shared_data[y] = (best_cluster == cluster) ? y_value : 0;
    shared_data[count] = (best_cluster == cluster) ? 1 : 0;
    __syncthreads();

    // Reduction for this cluster.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (local_index < stride) {
        shared_data[x] += shared_data[x + stride];
        shared_data[y] += shared_data[y + stride];
        shared_data[count] += shared_data[count + stride];
      }
      __syncthreads();
    }

    // Now shared_data[0] holds the sum for x.

    if (local_index == 0) {
      const int cluster_index = blockIdx.x * k + cluster;
      new_sums_x[cluster_index] = shared_data[x];
      new_sums_y[cluster_index] = shared_data[y];
      counts[cluster_index] = shared_data[count];
    }
    __syncthreads();
  }
}

__global__ void coarse_reduce(double* __restrict__ means_x,
                              double* __restrict__ means_y,
                              double* __restrict__ new_sum_x,
                              double* __restrict__ new_sum_y,
                              int k,
                              int* __restrict__ counts) {
  extern __shared__ double shared_data[];

  const int index = threadIdx.x;
  const int y_offset = blockDim.x;

  shared_data[index] = new_sum_x[index];
  shared_data[y_offset + index] = new_sum_y[index];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= k; stride /= 2) {
    if (index < stride) {
      shared_data[index] += shared_data[index + stride];
      shared_data[y_offset + index] += shared_data[y_offset + index + stride];
    }
    __syncthreads();
  }

  if (index < k) {
    const int count = max(1, counts[index]);
    means_x[index] = new_sum_x[index] / count;
    means_y[index] = new_sum_y[index] / count;
    new_sum_y[index] = 0;
    new_sum_x[index] = 0;
    counts[index] = 0;
  }
}

int main(int argc, const char* argv[]) {
  if (argc < 4) {
    // std::cerr << "usage: k-means <n> <k> [iterations]" << std::endl;
    std::cerr << "usage: k-means <file> <k> [iterations]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // const auto n = std::atoi(argv[1]); // total number of points
  const auto k = std::atoi(argv[2]); // number of centroids
  const auto number_of_iterations = std::atoi(argv[3]);

  // std::cout << "Total number of point:     " << n << std::endl;
  std::cout << "Total number of centroids: " << k << std::endl;
  std::cout << "Total number of iterations " << number_of_iterations << std::endl;

  // random generator
  std::mt19937 rng(std::random_device{}());
  // std::uniform_real_distribution<double> dist(-n, n);

  // TODO: generate data points on runtime, not from reading file
  std::vector<double> h_x; // contains all of the points
  std::vector<double> h_y; // contains all of the points
  // for (int i=0; i<n; i++) { // generates n points
  //   double x = dist(rng);
  //   double y = dist(rng);
  //   h_x.push_back(x); // point x
  //   h_y.push_back(y); // point y
  // }

  // TEST: run on multi-gpu vs run on single-gpu
  std::ifstream stream(argv[1]); // data file path
  std::string line;
  while (std::getline(stream, line)) {
    std::istringstream line_stream(line);
    float x, y;
    uint16_t label;
    line_stream >> x >> y >> label;
    h_x.push_back(x); // point x
    h_y.push_back(y); // point y
  }

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "There is no GPU in the system, exit" << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    std::cout << "Number of GPUs: " << device_count << std::endl;
  }

  Data d_means(k, h_x, h_y); // centroids from points on managed memory,
                             // which will be shared across devices.

  // shuffle points randomly
  std::shuffle(h_x.begin(), h_x.end(), rng);
  std::shuffle(h_y.begin(), h_y.end(), rng);

  size_t number_of_elements[device_count];
  int d_counts[device_count];
  Data d_data[device_count];
  Data d_sums[device_count];
  
  const int threads = 1024;
  int blocks[device_count];

  // * 3 for x, y and counts.
  const int fine_shared_memory = 3 * threads * sizeof(double);
  // * 2 for x and y. Will have k * blocks threads for the coarse reduction.
  int coarse_shared_memory[device_count];
  
  const auto start = std::chrono::high_resolution_clock::now();
  for (int device = 0; device < device_count; device++) {
    number_of_elements[device] = h_x.size() / device_count; // split points to each device
    blocks[device] = (number_of_elements[device] + threads - 1) / threads;

    coarse_shared_memory[device] = 2 * k * blocks[device] * sizeof(double);
    
    d_data[device] = Data(number_of_elements[device], h_x, h_y); // points on managed memory
    d_sums[device] = Data(k * blocks[device]); // sum of distance value from each block
  }

  for (int device = 0; device < device_count; device++) {
    cudaMallocManaged((void**)&d_counts[device], k * blocks[device] * sizeof(int));
    cudaMemset(&d_counts[device], 0, k * blocks[device] * sizeof(int));
  }

  std::cout << "Processing " << number_of_elements[0] << " points on " << blocks[0]
        << " blocks x " << threads << " threads" << std::endl;

  // do fine_reduce and coarse_reduce from each device
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    
    // for each of centroid, distance are calculated and saved to d_sum
    for (int device = 0; device < device_count; device++) {
      fine_reduce<<<blocks[device], threads, fine_shared_memory>>>(
                                                          d_data[device].x,
                                                          d_data[device].y,
                                                          d_data[device].size,
                                                          d_means.x,
                                                          d_means.y,
                                                          d_sums[device].x,
                                                          d_sums[device].y,
                                                          k,
                                                          d_counts);
    }
    cudaDeviceSynchronize();

    for (int device = 0; device < device_count; device++) {
      coarse_reduce<<<1, k * blocks[device], coarse_shared_memory[device]>>>(
                                                            d_means.x,
                                                            d_means.y,
                                                            d_sums[device].x,
                                                            d_sums[device].y,
                                                            k,
                                                            d_counts);
    }
    cudaDeviceSynchronize();

    // sum up distance from each devices
    std::vector<double> host_sums_x[device_count], host_sums_y[device_count];
    std::vector<int> host_counts[device_count];

    for (int device = 0; device < device_count; device++) {
      host_sums_x[device] = std::vector<double>(k * blocks[device]);
      host_sums_y[device] = std::vector<double>(k * blocks[device]);
      host_counts[device] = std::vector<int>(k * blocks[device]);

      memset(&host_sums_x[device], 0, sizeof(double) * k * blocks[device]);
      memset(&host_sums_y[device], 0, sizeof(double) * k * blocks[device]);
      memset(&host_counts[device], 0, sizeof(int) * k * blocks[device]);
    }

    for (int device = 0; device < device_count; device++) {
      memcpy(host_sums_x[device].data(), 
            d_sums[device].x, 
            d_sums[device].bytes);
      std::cout << "TEST 1" << std::endl;
      memcpy(host_sums_y[device].data(), 
            d_sums[device].y, 
            d_sums[device].bytes);
      std::cout << "TEST 2" << std::endl;
      memcpy(host_counts[device].data(), 
            &d_counts[device],
            k * blocks[device] * sizeof(int));
      std::cout << "TEST 3" << std::endl;
    }

    for (int cluster = 0; cluster < k; cluster++) {
      double total_sum_x = 0;
      double total_sum_y = 0;
      int total_count = 0;
      
      for (int device = 0; device < device_count; device++) {
        std::vector<double> vx = host_sums_x[device];
        std::vector<double> vy = host_sums_y[device];
        std::vector<int> vc = host_counts[device];

        total_sum_x += std::accumulate(vx.begin(), vx.end(), 0);
        total_sum_y += std::accumulate(vy.begin(), vy.end(), 0);
        total_count += std::accumulate(vc.begin(), vc.end(), 0);
      }
      
      if (total_count > 0) {
        d_means.x[cluster] = total_sum_x / total_count;
        d_means.y[cluster] = total_sum_y / total_count;
      }
    }
  }
  cudaFree(d_counts);

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
    std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
  std::cerr << "Standard CUDA implementation Took: " << duration.count() << "s" << " for "<<h_x.size()<<" points."<<std::endl;
  std_time_used = duration.count();

  std::vector<double> mean_x(k, 0); // host memory
  std::vector<double> mean_y(k, 0); // host memory
  // cudaMemcpy(mean_x.data(), d_means.x, d_means.bytes, cudaMemcpyDeviceToHost);
  // cudaMemcpy(mean_y.data(), d_means.y, d_means.bytes, cudaMemcpyDeviceToHost);
  memcpy(mean_x.data(), d_means.x, d_means.bytes);
  memcpy(mean_y.data(), d_means.y, d_means.bytes);

  for (size_t cluster = 0; cluster < k; ++cluster) {
    //std::cout << mean_x[cluster] << " " << mean_y[cluster] << std::endl;
  }

  FILE *fp;
  int i;

  fp = fopen("Standardtimes.txt", "a");
  fprintf(fp, "%0.6f\n", std_time_used);
  fclose(fp);

	
  std::string str(std::to_string(h_x.size())),str1,str2;
  str = "results/standard/" + str;

  str2 = str + "_centroids_mgpu.txt";
  fp = fopen(str2.c_str(), "w");
  for(i = 0; i < k; ++i){
    fprintf(fp, "%0.6f %0.6f\n", mean_x[i], mean_y[i]);
  }
  fclose(fp);
  
}
