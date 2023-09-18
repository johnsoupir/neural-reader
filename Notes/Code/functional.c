#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define TRAING_DATA_PATH "image.idx"
#define N 784
#define T 3

float  EPS = 0.01;		//Learning rate
float const ERROR = 0.03;	//Target ERROR
float output=0;
float miss=0;
float epochMiss = 0;


// Allocate memory for B, R, 
double Y[N];
double R[N][N][T+1];
double B[N][T+1];
double X[N][T+1];
double Z[N][T+1];
double dB[N][T+1];

double cost = ERROR + 1.0;
uint64_t mu, nu, t=0, cycles=0, epochs=0;


//Sigma function
double sigma(double input)
{
	return (1/(1+exp(-input)));
}

//Writes R to file
void write3DArrayToFile(uint8_t array[N][N][T+1], const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fwrite(array, sizeof(uint8_t), N * N * (T+1), file);
    fclose(file);
}

//Writes B to file
void write2DArrayToFile(uint8_t array[N][T+1], const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fwrite(array, sizeof(uint8_t), N * (T+1), file);
    fclose(file);
}


//Prints 3D array to screen
void print3DArray(double array[N][N][T+1]) {
    for (uint32_t x = 0; x < N; x++) {
        printf("Layer %u:\n", x);
        for (uint32_t y = 0; y < N; y++) {
            for (uint32_t z = 0; z < T+1; z++) {
                printf("%0.3f ", array[x][y][z]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


//Prints 2D array to screen
void print2DArray(double array[N][T+1]) {
    for (uint32_t x = 0; x < N; x++) {
        for (uint32_t y = 0; y < T+1; y++) {
            printf("%0.3f ", array[x][y]);
        }
        printf("\n");
    }
}

//Fill 3D array with random numbers -0.5 to 0.5
void seed3D(double array[N][N][T+1])
{
    for (uint32_t x = 0; x < N; x++) {
        for (uint32_t y = 0; y < N; y++) {
            for (uint32_t z = 0; z < T+1; z++) {
                array[x][y][z] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;//(double) (rand() % 1000) / 1000.0;
            }
        }
    }
}

//Fill 2D array with random numbers -0.5 to 0.5
void seed2D(double array[N][T+1])
{
    for (uint32_t y = 0; y < N; y++) {
        for (uint32_t x = 0; x < T+1; x++) {
            array[y][x] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;// (double) (rand() % 1000) / 1000.0;
        }
    }
}


int main()
{
    /*     LOAD IMAGE DATA     */
    FILE * trainingData = fopen("./image.idx", "rb");

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
    FILE * labelData = fopen("./labels.idx","rb");
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
	printf("Seeding B and R randomly...\n");
	seed3D(R);
	seed2D(B);


	/*    PRINTING IMAGES TO SCREEN    */
    printf("__________PRINTING TEST IMAGES TO SCREEN________\n");
    uint32_t chosenImage = 1;
    for(int blah=0; blah < 10; blah++)
    {
        chosenImage=blah;
        for(uint8_t printRow = 0; printRow < imageRows; printRow++)
        {
            // printf("\n");
            for(uint8_t printCol = 0; printCol < imageCols; printCol++)
            {
					// printf("Value %d at %d %d %d ", image[chosenImage][printRow][printCol], chosenImage, printCol, printRow);///255.00;
                if (image[chosenImage][printRow][printCol] > 128)
                {
                    printf("▇\n");
                }
                else
                {
                    printf(" \n");
                }


            }
            printf("\n");
        }
        // printf("The solution is %d\n", labels[chosenImage]);
    }


	
	//imageCount=100;
	// cost=ERROR+10;
	// while(epochs < 100)
	while( cost > ERROR )
	{
		for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
		{
			// Main training loop
		    // printf("\rLoaded sample %d into %d neurons.", imageIndex, mu);
		    // printf("\n Training epoch %ld >> Sample %d\tError %6.0f\tCycle %ld",epochs, imageIndex, cost, cycles);


			//Fill starting X with samples, Y with labels
			//Loop through every cell in the current sample, loading the image into the zeroth layer of the network and the label into the desired output.
			//Set mu to zero
			mu=0;
			for (int rowIndex=0; rowIndex<imageRows; rowIndex++)
			{
				for (int columnIndex=0; columnIndex<imageCols; columnIndex++)
				{
					X[mu][0] = (double)image[imageIndex][rowIndex][columnIndex]/255.00;

 				    /*
					if (image[imageIndex][rowIndex][columnIndex] > 127)
					{
						X[mu][0]=1;
					}
					else
					{
						X[mu][0]=0;
					}*/

					//X[mu][0] = (double)image[imageIndex][rowIndex][columnIndex]/255.00;

					// printf("X-Value %d at %d %d %d\n ", image[imageIndex][rowIndex][columnIndex], imageIndex, columnIndex, rowIndex);///255.00;

					Y[mu] = (double)labels[imageIndex]/10.0;
			        // printf("\nNeuron %ld=%f, L=%f ",mu, X[mu][0], Y[mu]);
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
					}

					X[mu][t] = sigma(Z[mu][t]);
				}
					
			}

			// back propagate
			// k=0 layer
			for (mu = 0; mu < N; mu++){
				dB[mu][T] = -EPS*(X[mu][T]-Y[mu])*X[mu][T]*(1-X[mu][T]);
				B[mu][T] = B[mu][T] + dB[mu][T];
				for (nu = 0; nu < N; nu++){
					R[mu][nu][T] = R[mu][nu][T] + dB[mu][T]*X[nu][T-1];
				}
			}

			// k = 1...T-1 layers
			for (int k = 1; k < T; k++){
				for (mu = 0; mu < N; mu++){
					dB[mu][T-k] = 0;
					for (int a = 0; a < N; a++){
						dB[mu][T-k] = dB[mu][T-k] + dB[a][T-k+1]*R[a][mu][T-k+1]*X[mu][T-k]*(1-X[mu][T-k]);
					}
					B[mu][T-k] = B[mu][T-k] + dB[mu][T-k];
					for (nu = 0; nu < N; nu++){
						R[mu][nu][T-k] = R[mu][nu][T-k] + dB[mu][T-k]*X[nu][T-k-1];
					}
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

			// printf("Guessed %f, answer %d, miss of %f", output, labels[imageIndex], miss );
			

			// calculate cost function
			cost = 0;
			for (mu = 0; mu < N; mu++){
				cost = cost + (X[mu][T]-Y[mu])*(X[mu][T]-Y[mu]);
			}
			cost = 0.5*cost;
			// increment number of back propagations
			cycles++;



		}
		
		//Calculate average error across last epoch
		printf("\n\n>>>>>>>>>>> EPOCH %d MISS AVERAGE: %f ",epochs, epochMiss/(float)imageCount);
		epochMiss=0;
		epochs++;
		if (epochs > 50)
		{
			EPS = 0.001;
		}
	}//End training

	//Training is done. Now save the parameters to file.
	write2DArrayToFile(B,"B.neural");
	write3DArrayToFile(R,"R.neural");
	
	//Now we test!
	printf("We've come so far, and and tried so hard. \n In the end: \n");

	for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
	{
		   // printf("\rLoaded sample %d into %d neurons.", imageIndex, mu);
		   printf("\n Training epoch %d >> Sample %d\tError %6.5f\tCycle %d",epochs, imageIndex, cost, cycles);


		//Fill starting X with samples, Y with labels
		//Loop through every cell in the current sample, loading the image into the zeroth layer of the network and the label into the desired output.
		//Set mu to zero
		mu=0;
		for (int rowIndex=0; rowIndex<imageRows; rowIndex++)
		{
			for (int columnIndex=0; columnIndex<imageCols; columnIndex++)
			{
				X[mu][0] = (double)image[imageIndex][rowIndex][columnIndex];
				Y[mu] = (double)labels[imageIndex];
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
				}

				X[mu][t] = sigma(Z[mu][t]);
			}
				
		}


		printf("For sample %d, the result is ", imageIndex);
		float sum=0;

		for (int i = 0; i < N; i++)
		{
			sum += X[i][T];
			if (X[i][T] != 1.00)
			{
				printf("END-Neuron %d=%f ",i, X[i][T]);
				/* code */
			}
			
		}
		float output = sum/784.0;
		printf("%f. And the answer is %d/n",output, labels[imageIndex]);


	}//End testing

}//End main

