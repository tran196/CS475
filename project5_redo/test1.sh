
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=1024
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=128000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=512000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=896000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=1280000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=1536000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=3968000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=4352000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=6400000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=7168000
./arrayMult
g++ -o arrayMult arrayMult.cpp libOpenCL.so -lm -fopenmp -D LOCAL_SIZE=8 -D NUM_ELEMENTS=8192000
./arrayMult
