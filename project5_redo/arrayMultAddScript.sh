#!/bin/bash

echo "------------------- globalWorkSize - Array Multiply ---------------------------"
for i in 32 64 128 256 512
do 
    for j in 1024 128000 512000 896000 1280000 1536000 3968000 4352000 6400000 7168000 8192000 
    do 
        g++ -o arrayMultAdd arrayMultAdd.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=$i -D NUM_ELEMENTS=$j;
        ./arrayMultAdd 2>>resultArrayMultAdd;
    done 
    i=$((i=i + 1)); 
done