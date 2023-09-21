#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "arrayTools.h"

#define TESTING_DATA_PATH "TestData/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
#define TESTING_LABEL_PATH "TestData/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"

#define B_PARAMETER_PATH "Parameters/B.neural"
#define R_PARAMETER_PATH "Parameters/R.neural"

#define N 784
#define T 3

float output=0;
float miss=0;
float epochMiss = 0;
float epochMissAverage = 0;


// Allocate memory for B, R, 
double Y[N];
double R[N][N][T+1];
double B[N][T+1];
double X[N][T+1];
double Z[N][T+1];
uint64_t mu, nu, t=0, cycles=0, epochs=0;


//Sigma function
double sigma(double input)
{
	return (1/(1+exp(-input)));
}


int main()
{
    /*     LOAD IMAGE DATA     */
    FILE * trainingData = fopen(TESTING_DATA_PATH, "rb");

    if (trainingData == NULL)
    {
        printf("\nERROR OPENING TRAINING DATA!!!\n");
    }
    else
    {
        printf("Opened training data.\n");
    }

    uint32_t magicNumber, imageCount, imageRows, imageCols;
    fread(&magicNumber, 4, 1, trainingData);
    fread(&imageCount, 4, 1, trainingData);
    fread(&imageCols, 4, 1, trainingData);
    fread(&imageRows, 4, 1, trainingData);
    magicNumber = __builtin_bswap32(magicNumber);
    imageCount = __builtin_bswap32(imageCount);
    imageCols = __builtin_bswap32(imageCols);
    imageRows = __builtin_bswap32(imageRows);

    printf("The training data contains:\n  %d images\n  Resolution %d x %d\n", imageCount, imageCols, imageRows);
    printf("Allocating %d bytes of memory.\n", imageCols*imageCols*imageCount);
    uint8_t (*image)[imageRows][imageCols] = malloc(imageCount * sizeof(*image));

    printf("Loading data into memory...\n");
    for(uint32_t imageIndex=0; imageIndex < imageCount; imageIndex++ )
    {
        for(uint32_t rowIndex=0; rowIndex < imageRows; rowIndex++)
        {
            for(uint32_t columIndex=0; columIndex < imageCols; columIndex++)
            {
                fread(&image[imageIndex][rowIndex][columIndex], 1, 1, trainingData);
            }
        }
    }
    printf("DONE\n");


    /*     LOAD LABEL DATA     */
    FILE * labelData = fopen(TESTING_LABEL_PATH,"rb");
    if (labelData == NULL)
    {
        printf("\nERROR OPENING LABEL DATA!!!\n");
    }
    else
    {
        printf("Opened label data.\n");
    }
    uint32_t label_magicNumber, labelCount;
    fread(&label_magicNumber, 4, 1, labelData);
    fread(&labelCount, 4, 1, labelData);
    label_magicNumber = __builtin_bswap32(label_magicNumber);
    labelCount = __builtin_bswap32(labelCount);

    printf("The label data contains:\n  %d labels\n", labelCount);
    printf("Alloc label mem\n");
    uint8_t labels[labelCount];
    for(uint32_t labelIndex = 0; labelIndex < labelCount; labelIndex++)
    {
        fread(&labels[labelIndex], 1, 1, labelData);
    }


	/* Seed the B and R arrays with random numbers */
	printf("Loading B and R from file...\n");
	read2DArrayFromFile(B,B_PARAMETER_PATH);
	read3DArrayFromFile(R,R_PARAMETER_PATH);

	printf("---> Reading started! <----");
	
	for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
	{
		mu=0;
		for (int rowIndex=0; rowIndex<imageRows; rowIndex++)
		{
			for (int columnIndex=0; columnIndex<imageCols; columnIndex++)
			{
				X[mu][0] = (double)image[imageIndex][rowIndex][columnIndex]/255.00;
				Y[mu] = (double)labels[imageIndex]/10.0;
				mu++;
			}
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
					//This can be +=
				}

				X[mu][t] = sigma(Z[mu][t]);
			}
				
		}

		float sum=0;
		for (int mu = 0; mu < N; mu++)
		{
			sum += X[mu][T];
		}
		output = sum/784.0;
		miss = fabs(((float)labels[imageIndex]/10) - output);
		epochMiss += miss;

		if (imageIndex % 1000 == 0)
		{
			printf("\nSample %d -> Guessed %1.5f, answer %d, miss of %1.5f",imageIndex, output*10, labels[imageIndex], miss*10 );
		}
		cycles++;
	}
}//End main

