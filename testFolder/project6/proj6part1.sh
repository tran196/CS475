#!/bin/csh

for local_size in 8 16 32 64 128 256 512
do
	for NUM in 1024 4096 8192 32768 65536 131072 262144 524288 1048576 2097152 3145728 4194304 5242880 6291456 7340032 8388608
	do
		g++ -o proj6part1 proj6part1.cpp /scratch/cuda-7.0/lib64/libOpenCL.so -lm -fopenmp -DNUM=$NUM -DLOCAL_SIZE=$local_size -DMULT
		./proj6part1

		g++ -o proj6part1 proj6part1.cpp /scratch/cuda-7.0/lib64/libOpenCL.so -lm -fopenmp -DNUM=$NUM -DLOCAL_SIZE=$local_size -DMULT_ADD
		./proj6part1
	done
done
