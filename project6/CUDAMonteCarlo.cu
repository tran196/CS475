#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE		64		// number of threads per block
#endif

// setting the number of trials in the monte carlo simulation:
#ifndef NUMTRIALS
#define NUMTRIALS	(1024*1024)
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif


// ranges for the random numbers:
const float XCMIN =	 0.0;
const float XCMAX =	 2.0;
const float YCMIN =	 0.0;
const float YCMAX =	 2.0;
const float RMIN  =	 0.5;
const float RMAX  =	 2.0;

// function prototypes:
float		Ranf( float, float );
int			Ranf( int, int );
void		TimeOfDaySeed( );


// array multiplication (CUDA Kernel) on the device: C = A * B

__global__  void MonteCarlo( float *X, float *Y, float *R , int *N)
{
	__shared__ float prods[BLOCKSIZE];

	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	int numHits = 0;
	
				// randomize the location and radius of the circle:
				// float xc = xcs[n];
				// float yc = ycs[n];
				// float  r =  rs[n];

				float xc = X[gid];
				float yc = Y[gid];
				float  r =  R[gid];

				// solve for the intersection using the quadratic formula:
				float a = 2.;
				float b = -2.*( xc + yc );
				float c = xc*xc + yc*yc - r*r;
				float d = b*b - 4.*a*c;

				// If d is less than 0., then the circle was completely missed. (Case A) Continue on to the next trial in the for-loop.
				if (d >= 0.)
				{
				
					// hits the circle:
					// get the first intersection:
					d = sqrt( d );
					float t1 = (-b + d ) / ( 2.*a );	// time to intersect the circle
					float t2 = (-b - d ) / ( 2.*a );	// time to intersect the circle
					float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection
				
					// If tmin is less than 0., then the circle completely engulfs the laser pointer. (Case B) Continue on to the next trial in the for-loop.
					if (tmin >= 0.)
					{

						// where does it intersect the circle?
						float xcir = tmin;
						float ycir = tmin;

						// get the unitized normal vector at the point of intersection:
						float nx = xcir - xc;
						float ny = ycir - yc;
						float n = sqrt( nx*nx + ny*ny );
						nx /= n;	// unit vector
						ny /= n;	// unit vector

						// get the unitized incoming vector:
						float inx = xcir - 0.;
						float iny = ycir - 0.;
						float in = sqrt( inx*inx + iny*iny );
						inx /= in;	// unit vector
						iny /= in;	// unit vector

						// get the outgoing (bounced) vector:
						float dot = inx*nx + iny*ny;
						float outx = inx - 2.*nx*dot;	// angle of reflection = angle of incidence`
						float outy = iny - 2.*ny*dot;	// angle of reflection = angle of incidence`

						// find out if it hits the infinite plate:
						float t = ( 0. - ycir ) / outy;
						// If t is less than 0., then the reflected beam went up instead of down. Continue on to the next trial in the for-loop.

						if(t >= 0.)
						{
							
							// Otherwise, this beam hit the infinite plate. (Case D) Increment the number of hits and continue on to the next trial in the for-loop.
							numHits++;
						}
					}

				}
	N[gid] = numHits;
}


// main program:

int
main( int argc, char* argv[ ] )
{
	TimeOfDaySeed( );		// seed the random number generator

	int dev = findCudaDevice(argc, (const char **)argv);

	// allocate host memory:

	// From OpenCL MonteCarlo
	// better to define these here so that the rand() calls don't get into the thread timing:
	float *xcs = new float [NUMTRIALS];
	float *ycs = new float [NUMTRIALS];
	float *rs = new float [NUMTRIALS];

	int *n = new int [NUMTRIALS];

	// fill the random-value arrays:
	for( int n = 0; n < NUMTRIALS; n++ )
	{       
			xcs[n] = Ranf( XCMIN, XCMAX );
			ycs[n] = Ranf( YCMIN, YCMAX );
			rs[n] = Ranf(  RMIN,  RMAX ); 
	}    
	// allocate device memory:
	float *dXCS, *dYCS, *dRS;
	int * dN;

	dim3 dimsXCS( NUMTRIALS, 1, 1 );
	dim3 dimsYCS( NUMTRIALS, 1, 1 );
	dim3 dimsRS( NUMTRIALS, 1, 1 );
	dim3 dimsN (NUMTRIALS, 1, 1);

	//__shared__ float prods[SIZE/BLOCKSIZE];

	cudaError_t status;
	status = cudaMalloc( reinterpret_cast<void **>(&dXCS), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dYCS), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dRS), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dN), NUMTRIALS*sizeof(int) );
		checkCudaErrors( status );

	// copy host memory to the device:

	status = cudaMemcpy( dXCS, xcs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
	checkCudaErrors( status );
	status = cudaMemcpy( dYCS, ycs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
	checkCudaErrors( status );
	status = cudaMemcpy( dRS, rs, (NUMTRIALS)*sizeof(float), cudaMemcpyHostToDevice );
	checkCudaErrors( status );

	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid( NUMTRIALS / threads.x, 1, 1 );

	// Create and start timer

	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );

	// execute the kernel:
	
	MonteCarlo<<< grid, threads >>>( dXCS, dYCS, dRS , dN);

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
		checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );

	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double multsPerSecond = (float)NUMTRIALS / secondsTotal;
	double megaMultsPerSecond = multsPerSecond / 1000000.;
	fprintf( stderr, "Array Size = %10d, MegaMultTrials/Second = %10.2lf\n", NUMTRIALS, megaMultsPerSecond );

	fprintf( stderr, "\nBLOCKSIZE Size = %d, NUMTRIALS = %d\n", BLOCKSIZE, NUMTRIALS );

	// copy result from the device to the host:
	
	status = cudaMemcpy( n, dN, (NUMTRIALS)*sizeof(int), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

	int numHits = 0;
	for(int i = 0; i < NUMTRIALS; i++ )
	{
		numHits += (int)n[i];
	}
	fprintf( stderr, "\nnumHits = %d\n", numHits );

	printf("\nProbability: %f\n", (float)numHits/(float)NUMTRIALS);

	// clean up memory:
	delete [ ] xcs;
	delete [ ] ycs;
	delete [ ] rs;
	delete [ ] n;

	status = cudaFree( dXCS );
		checkCudaErrors( status );
	status = cudaFree( dYCS );
		checkCudaErrors( status );
	status = cudaFree( dRS );
		checkCudaErrors( status );
	status = cudaFree( dN );
		checkCudaErrors( status );

	return 0;
}


//Helper Functions
float
Ranf( float low, float high )
{
        float r = (float) rand();               // 0 - RAND_MAX
        float t = r  /  (float) RAND_MAX;       // 0. - 1.

        return   low  +  t * ( high - low );
}

int
Ranf( int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = ceil( (float)ihigh );

        return (int) Ranf(low,high);
}

void
TimeOfDaySeed( )
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time( &timer );
	double seconds = difftime( timer, mktime(&y2k) );
	unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
	srand( seed );
}
