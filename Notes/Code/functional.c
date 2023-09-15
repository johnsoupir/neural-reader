#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>



float const EPS = 0.2;		//Learning rate
float const ERROR = 0.0001;	//Target ERROR


// Allocate memory for B, R, 

#define ERROR 0.0001
#define EPS 0.2
#define N 784
#define T 20


uint8_t INPUT[4][2] = {
	{1,1},
	{1,0},
	{0,1},
	{0,0}
};

uint8_t OUTPUT[4] = {1,0,0,0};

uint8_t Y[N];
uint8_t R[N][N][T+1];
uint8_t B[N][T+1];
uint8_t X[N][T+1];
uint8_t Z[N][T+1];
uint8_t dB[N][T+1];

double cost = ERROR + 1;

int mu, nu, t=0, cycles=0;


//Sigma function
double sigma(double input)
{
	return (1/(1+exp(-input)));
}




void seedRandom(uint8_t * array, uint32_t rows, uint32_t cols, uint32_t layers)
{
	srand(time(NULL));

	if(layers > 1)
	{
		//3D Array
		for(uint32_t layerIndex=0; layerIndex<layers; layerIndex++)
		{
			for(uint32_t columnIndex=0; columnIndex<cols; columnIndex++)
			{
				for(uint32_t rowIndex=0; rowIndex<rows; rowIndex++)
				{
					//FILL ARRAY
//					*((arr + i*y*z) + (j*z) + k) = 0;
//					*((arr + i*cols) + j) = (rand()%1000)/1000.0;
	
				}
			}
		}
	}
	else
	{
		//2D array
		for(uint32_t columnIndex=0; columnIndex<=cols; columnIndex++)
		{
			for(uint32_t rowIndex=0; rowIndex<rows; rowIndex++)
			{
				//FILL ARRAY
				*((array + columnIndex*cols) + rowIndex) = (rand()%1000)/1.0;

			}
		}
	}
}






int main()
{
	for(int i=0; i<4; i++)
	{
		for(int j=0; j<2; j++)
		{
			printf("%d\t",INPUT[i][j]);
		}
		printf("\n");
	}
	
	printf("Seeding random\n");

	seedRandom((uint8_t *)INPUT,4,2,1);
	
	while( cost > ERROR )
	{
		for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
		{
			// Main training loop


			//Fill starting X with samples, Y with labels
			for (mu = 0; mu < N; mu++)
			{
				X[mu][0] = IN[sampleIndex][mu];
				Y[mu]    = OUT[sampleIndex][0];

			}


			//Forward propogate
			for (t=1; t <= T; t++)
			{
				for (mu = 0; mu < N; mu++)
				{
					Z[mu][t] = B[mu][t];
					
					for (nu = 0; nu < N; nu++)
					{
						Z[mu][t] = Z[mu][t] + R[mu][nu][t]*X[nu][t-1];
					}

					X[mu][t] = sigma(Z[mu][t]);
				}
					
			}




		}
	}
	




//WHILE LOOP -> NOT TRAINED

	//FOR -> TRAINING SAMPLES
	
		//Load training sample into input layer
		
		//Forward propogate
		
		//Back propogate
		
		//Tune network
		
		//Get cost
		
		
	




	for(int i=0; i<4; i++)
	{
		for(int j=0; j<2; j++)
		{
			printf("%d\t",INPUT[i][j]);
		}
		printf("\n");
	}


}

