#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>



float const EPS = 0.2;		//Learning rate
float const ERROR = 0.0001;	//Target ERROR


uint8_t INPUT[4][2] = {
	{1,1},
	{1,0},
	{0,1},
	{0,0}
};



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

	for(int i=0; i<4; i++)
	{
		for(int j=0; j<2; j++)
		{
			printf("%d\t",INPUT[i][j]);
		}
		printf("\n");
	}


}

