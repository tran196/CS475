#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "simd.p4.h"

#define NUMT        1
#define ARRAYSIZE   100000    //you decide
#define NUMTRIES    10    	// you decide

float A[ARRAYSIZE];
float B[ARRAYSIZE];
float C[ARRAYSIZE];

int main( )
{

#ifndef _OPENMP
        fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
        return 1;
#endif

        omp_set_num_threads( NUMT );
        fprintf( stderr, "Using %d threads\n", NUMT );

        double maxMegaMults = 0.;
	double executionTime = 0.;

        for( int t = 0; t < NUMTRIES; t++ )
        {
                double time0 = omp_get_wtime( );

                #pragma omp parallel for
                for( int i = 0; i < ARRAYSIZE; i++ )
                {
                        C[i] = A[i] * B[i];
                }

                double time1 = omp_get_wtime( );
                double megaMults = (double)ARRAYSIZE/(time1-time0)/1000000.;
                if( megaMults > maxMegaMults )
                        maxMegaMults = megaMults;
		executionTime = time1 - time0;
        }

        printf( "Peak Performance = %8.2lf MegaMults/Sec\n", maxMegaMults );
        printf( "Execution time for %d threads: %lf\n", NUMT, executionTime );
	// note: %lf stands for "long float", which is how printf prints a "double"
	//        %d stands for "decimal integer", not "double"


        omp_set_num_threads( NUMT );
        fprintf( stderr, "\nSIMD Results Using %d threads\n", NUMT );

        double simd_maxMegaMults = 0.;
	double simd_executionTime = 0.;

        for( int t = 0; t < NUMTRIES; t++ )
        {
                double simd_time0 = omp_get_wtime( );

                // #pragma omp parallel for
                // for( int i = 0; i < ARRAYSIZE; i++ )
                // {
                //         C[i] = A[i] * B[i];
                // }

                SimdMul(A, B, C, ARRAYSIZE);

                double simd_time1 = omp_get_wtime( );
                double simd_megaMults = (double)ARRAYSIZE/(simd_time1-simd_time0)/1000000.;
                if( simd_megaMults > simd_maxMegaMults )
                        simd_maxMegaMults = simd_megaMults;
		simd_executionTime = simd_time1 - simd_time0;
        }

        printf( "SIMD Peak Performance = %8.2lf MegaMults/Sec\n", simd_maxMegaMults );
        printf( "SIMD Execution time for %d threads: %lf\n", NUMT, simd_executionTime );
	// note: %lf stands for "long float", which is how printf prints a "double"
	//   

        return 0; 
}