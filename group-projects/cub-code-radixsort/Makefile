CUB=cub-1.8.0

all: cub-sort

cub-sort: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-cub sorting_test.cu
	./test-cub 100000000

clean:
	rm -f test-cub

