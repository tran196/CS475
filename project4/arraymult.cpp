#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "simd.p4.h"

#define NUMT        1

#ifndef ARRAYSIZE
#define ARRAYSIZE   100000    //you decide
#endif  

#define NUMTRIES    10    	// you decide

float A[ARRAYSIZE];
float B[ARRAYSIZE];
float C[ARRAYSIZE];

int main( )
{

//Array Multiplication Non-SIMD
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

//Array Multiplication SIMD
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



        //Array Multiplcation + Reduction Timing

        omp_set_num_threads( NUMT );
        fprintf( stderr, "\nArray Multiplcation + Reduction Timing Using %d threads\n", NUMT );

        maxMegaMults = 0.;
        executionTime = 0.;
        float sum = 0.;


        for( int t = 0; t < NUMTRIES; t++ )
        {
                double time0 = omp_get_wtime( );

                // #pragma omp parallel for
                // for( int i = 0; i < ARRAYSIZE; i++ )
                // {
                //         sum += A[i] * B[i];
                // }

                NonSimdMulSum(A, B, ARRAYSIZE);

                double time1 = omp_get_wtime( );
                double megaMults = (double)ARRAYSIZE/(time1-time0)/1000000.;
                if( megaMults > maxMegaMults )
                        maxMegaMults = megaMults;
		executionTime = time1 - time0;
        }

        printf( "Array Multiplcation + Reduction Peak Performance = %8.2lf MegaMults/Sec\n", maxMegaMults );
        printf( "Array Multiplcation + Reduction Execution time for %d threads: %lf\n", NUMT, executionTime );
	// note: %lf stands for "long float", which is how printf prints a "double"
	//        %d stands for "decimal integer", not "double"


        // // SIMD Array Multiplcation + Reduction

        omp_set_num_threads( NUMT );
        fprintf( stderr, "\nSIMD Array Multiplcation + Reduction Results Using %d threads\n", NUMT );

         simd_maxMegaMults = 0.;
	 simd_executionTime = 0.;
        float simd_sum[4] = { 0., 0., 0., 0. };

        for( int t = 0; t < NUMTRIES; t++ )
        {
                double simd_time0 = omp_get_wtime( );

                // #pragma omp parallel for
                // for( int i = 0; i < ARRAYSIZE; i++ )
                // {
                //         C[i] = A[i] * B[i];
                // }

        //        SimdMulSum(A, B, ARRAYSIZE);

                double simd_time1 = omp_get_wtime( );
                double simd_megaMults = (double)ARRAYSIZE/(simd_time1-simd_time0)/1000000.;
                if( simd_megaMults > simd_maxMegaMults )
                        simd_maxMegaMults = simd_megaMults;
		simd_executionTime = simd_time1 - simd_time0;
        }

        printf( "SIMD Array Multiplcation + Reduction Peak Performance = %8.2lf MegaMults/Sec\n", simd_maxMegaMults );
        printf( "SIMD Array Multiplcation + ReductionExecution time for %d threads: %lf\n", NUMT, simd_executionTime );
	// note: %lf stands for "long float", which is how printf prints a "double"
	//   

        return 0; 
}
