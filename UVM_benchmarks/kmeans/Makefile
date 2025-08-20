include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

all: kmeans_standard kmeans_standard_mgpu

TEST=1

.PHONY: clean all

kmeans_cuda:
	$(CC) -std=c++11 kmeans_cuda.cu -o kmeans_cuda

kmeans_standard:
	$(CC) -std=c++11 standard_kmeans.cu -o kmeans_standard 

kmeans_standard_mgpu:
	$(CC) -std=c++11 standard_kmeans_mgpu.cu -o kmeans_standard_mgpu -DTEST=$(TEST)

clean:
	rm -f kmeans_cuda kmeans_standard
