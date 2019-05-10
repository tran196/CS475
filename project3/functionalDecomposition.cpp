// Tuan Tran
// Project 3

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>

//State of the System
//Global Variables
int	NowYear;		// 2019 - 2024
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	NowNumDeer;		// number of deer in the current population
int AdditionalDeer;

//Basic Time Step Will be One Month
const float GRAIN_GROWS_PER_MONTH =		8.0;
const float ONE_DEER_EATS_PER_MONTH =		0.5;

const float AVG_PRECIP_PER_MONTH =		6.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				50.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

unsigned int seed;


omp_lock_t Lock;
int NumInThreadTeam;
int NumAtBarrier;
int NumGone;

//Functions

// function prototypes:
float		Ranf(unsigned int *seedp, float, float );
int			Ranf(unsigned int *seedp, int, int );
void		TimeOfDaySeed( );
float       SQR( float x );
void        GrainDeer();
void        Grain();
void        Watcher();
void        MoreDeer();


// Main

int main ()
{
    // TimeOfDaySeed();
    // starting date and time:
    NowMonth =    0;
    NowYear  = 2019;

    // starting state (feel free to change this if you want):
    NowNumDeer = 1;
    NowHeight =  1.;

// The temp and precipitation are a function of the particular month
// A year consists of 12 months and 30 days each
// First Day of winter is January 1
// Temp and Precipitation follow cosine and since Wave patterns with some randomness added
    // float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

    // float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    // unsigned int seed = 0;
    // NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

    // float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    // NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
    // if( NowPrecip < 0. )
    //     NowPrecip = 0.;


    printf( "Month\tYear\tTemp(C)\tPrecipitation(cm)\tGrain Height\tDeer\tAdditional Deer Added This Month\n");



    omp_set_num_threads( 4 );	// same as # of sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            GrainDeer( );
        }

        #pragma omp section
        {
            Grain( );
        }

        #pragma omp section
        {
            Watcher( );
        }

        #pragma omp section
        {
            MoreDeer( );	// your own
        }
    }       // implied barrier -- all functions must return in order
        // to allow any of them to get past here

    return 0;
}

void
GrainDeer( )
{
    int TempNumDeer;
    while( NowYear < 2025 )
    {
        TempNumDeer = NowNumDeer;
        if (NowNumDeer < NowHeight) 
        {
            TempNumDeer++;
        }
        else if (NowNumDeer > NowHeight)
        {
            TempNumDeer--;
        }

    // DoneComputing barrier:
	#pragma omp barrier

    NowNumDeer = TempNumDeer;
    

    // DoneAssigning barrier:
    #pragma omp barrier

    // DonePrinting barrier:
	#pragma omp barrier

    }
}

void
Grain( )
{

    float TempHeight;
    while( NowYear < 2025 )
    {
        TempHeight = NowHeight;

        float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );

        float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );

        TempHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        TempHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;

        if (TempHeight < 0)
        {
            TempHeight = 0.;
        }

     // DoneComputing barrier:
	#pragma omp barrier

        NowHeight = TempHeight;

    // DoneAssigning barrier:
    #pragma omp barrier

    // DonePrinting barrier:
	#pragma omp barrier

    }

}

void
Watcher( )
{
    int tempMonth;
    int tempYear;
    float tempTemp;
    float tempPrecip;
    unsigned int seed = 0;

    while( NowYear < 2025 )
    {

        // DoneComputing barrier:
        #pragma omp barrier
        // . . .


        // DoneAssigning barrier:
        #pragma omp barrier
        // . . .

        // printf( "   Month, %d\t  Year, %d\t    Temp(C), %.2f\t   Precipitation(cm), %.2f\t  Grain Height(cm), %.2f\t    Deer, %d\t Deer Eaten By Bear: %d\n", 
        //             NowMonth,   NowYear,    (5./9.) *(NowTemp-32),    NowPrecip*2.54,          NowHeight*2.54,          NowNumDeer , DeerEatenBear);

        // printf( "Month\tYear\tTemp(C)\tPrecipitation(cm)\tGrain Height\tDeer\tAdditional Deer Added This Month\n");
        printf( "%d\t%d\t%.2f\t%.2f\t%.2f\t%d\t%d\n", 
        NowMonth, NowYear, (5./9.)*(NowTemp-32),NowPrecip*2.54, NowHeight, NowNumDeer, AdditionalDeer);


        tempMonth = NowMonth + 1;
        tempYear = NowYear;

        if (tempMonth == 12)
        {
            tempYear++;
            tempMonth = 0;
        }

        NowMonth = tempMonth;
        NowYear = tempYear;

        //Recompute Temperature and Precipitation for Next Month
        float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

        float temp = AVG_TEMP - AMP_TEMP * cos( ang );
        
        NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
        NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
        if( NowPrecip < 0. )
            NowPrecip = 0.;

        // DonePrinting barrier:
        #pragma omp barrier
        // . . .
    }

}

void
MoreDeer( )
{
    int tempDeer;
    while( NowYear < 2025 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        // . . .
        tempDeer = NowNumDeer;
        if (NowMonth == 1)
        {
            // AdditionalDeer = 1;
            AdditionalDeer = 6;
        }
        else
        {
            AdditionalDeer = 0;
        }

        
        
        

        // DoneComputing barrier:
        #pragma omp barrier
        // . . .
        
        // tempDeer += AdditionalDeer;
        // NowNumDeer = tempDeer;
        NowNumDeer += AdditionalDeer;

        // DoneAssigning barrier:
        #pragma omp barrier
        // . . .
        
        // DonePrinting barrier:
        #pragma omp barrier
        // . . .
    }

}


void
InitBarrier( int n )
{
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock( &Lock );
}

void
WaitBarrier( )
{
    omp_set_lock( &Lock );
    {
        NumAtBarrier++;
        if( NumAtBarrier == NumInThreadTeam ) // release the waiting threads
        {
        NumGone = 0;
        NumAtBarrier = 0;
        // let all other threads return before this one unlocks:
        while( NumGone != NumInThreadTeam - 1 );
        omp_unset_lock( &Lock );
        return;
        }
    }
    omp_unset_lock( &Lock );
    while( NumAtBarrier != 0 ); // all threads wait here until the last one arrives
    #pragma omp atomic // and sets NumAtBarrier to 0
    NumGone++;

}

//Helper Functions
float
Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


int
Ranf( unsigned int *seedp, int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
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

float
SQR( float x )
{
        return x*x;
}