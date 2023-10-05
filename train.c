#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define N 784 //Number of neurons. MNIST has 28x28 pixel images, so N=784.
#define T 3   //Number of hidden layers. 

#define TRAINING_DATA_PATH "TrainingData/train-images-idx3-ubyte/train-images.idx3-ubyte"
#define TRAINING_LABEL_PATH "TrainingData/train-labels-idx1-ubyte/train-labels.idx1-ubyte"

#define B_PARAMETER_PATH "Parameters/B.neural"
#define R_PARAMETER_PATH "Parameters/R.neural"

#define B_PARAMETER_BACKUP "B_backup.neural"
#define R_PARAMETER_BACKUP "R_backup.neural"

bool oldDogNewTricks = false; 	//Continue training on top of existing parameters
bool resume = false; 			//Retore from backup and continue training

float  EPS = 0.01;				//Learning rate
float const ERROR = 0.001;		//Target ERROR

float output=0;
float miss=0;
float epochMiss = 0;
float epochMissAverage = 0;

clock_t start, epochStart, checkpointStart;
char formattedTime[9];
uint32_t secondsSinceStart=0;

//Declare arrays to hold neural net
double Y[N];
double R[N][N][T+1];
double B[N][T+1];
double X[N][T+1];
double Z[N][T+1];
double dB[N][T+1];

double cost = ERROR + 1.0;
uint64_t mu, nu, t=0, cycles=0, epochs=0;

//Sigma activation function. Returns the next X.
double sigma(double input)
{
	return (1/(1+exp(-input)));
}

//Tahn activation function.
double tanh(double input)
{
	return (2/(1+exp(-input*2)));
}

//Mish activation function
double mish(double input)
{
	return (input * tanh(log1p(exp(input))));
}

//Returns seconds since a given time
double timeSince(clock_t checkpoint) {
    clock_t current = clock();
    double elapsed_time = ((double) (current - checkpoint)) / CLOCKS_PER_SEC;
    return elapsed_time;
}

//Logs string to file
bool logToFile(const char *filename, const char *content) {
    FILE *logFile = fopen(filename, "a");
    if (logFile == NULL) {
        fprintf(stderr, "Could not open file %s for writing.\n", filename);
        return false;
    }
    
    fprintf(logFile, "%s\n", content);
    fclose(logFile);
    return true;
}

//Clears a given log file
bool clearLog(const char *filename) {
    if (remove(filename) == 0) {
        printf("%s cleared successfully.\n", filename);
        return true;
    } else {
        fprintf(stderr, "Error deleting the file %s.\n", filename);
        return false;
    }
}

// Writes a 3D array of doubles to a file
void write3DArrayToFile(double array[N][N][T+1], const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fwrite(array, sizeof(double), N * N * (T+1), file);
    fclose(file);
}

// Writes a 2D array of doubles to a file
void write2DArrayToFile(double array[N][T+1], const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fwrite(array, sizeof(double), N * (T+1), file);
    fclose(file);
}

// Reads a 3D array of doubles from a file
void read3DArrayFromFile(double array[N][N][T+1], const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fread(array, sizeof(double), N * N * (T+1), file);
    fclose(file);
}

// Reads a 2D array of doubles from a file
void read2DArrayFromFile(double array[N][T+1], const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fread(array, sizeof(double), N * (T+1), file);
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
                array[x][y][z] = (((double) rand() / RAND_MAX) - 0.5)/0.5;//(double) (rand() % 1000) / 1000.0;
            }
        }
    }
}

//Fill 2D array with random numbers -0.5 to 0.5
void seed2D(double array[N][T+1])
{
    for (uint32_t y = 0; y < N; y++) {
        for (uint32_t x = 0; x < T+1; x++) {
            array[y][x] = (((double) rand() / RAND_MAX) - 0.5)/0.5;// (double) (rand() % 1000) / 1000.0;
        }
    }
}



/*    ------  MAIN ------     */

int main()
{
	srand(time(NULL)); //Seed random generator with time
	start = clock(); //Set the start point for the timer

    /*     LOAD IMAGE DATA     */
    FILE * trainingData = fopen(TRAINING_DATA_PATH, "rb"); //Open training images
    if (trainingData == NULL) //Check that file opened, exit on failure.
    {
        printf("\nERROR OPENING TRAINING DATA!!!\n");
		return 0;
    }
    else
    {
        printf("Opened training data.\n");
    }

	//Declare unsigned 32 bit integer to hold information from the file.
    uint32_t magicNumber, imageCount, imageRows, imageCols;
    fread(&magicNumber, 4, 1, trainingData);	//Read the "magic number", which specifies the data type and how many dimensions
    fread(&imageCount, 4, 1, trainingData);		//Read now many images
    fread(&imageCols, 4, 1, trainingData);		//Read how many columns in each image
    fread(&imageRows, 4, 1, trainingData);		//Read how many rows in each image

	//File format is big-endian. Reverse the bytes into little-endian so we can read them.
    magicNumber = __builtin_bswap32(magicNumber);	
    imageCount = __builtin_bswap32(imageCount);
    imageCols = __builtin_bswap32(imageCols);
    imageRows = __builtin_bswap32(imageRows);

    printf("The training data contains:\n  %d images\n  Resolution %d x %d\n", imageCount, imageCols, imageRows);
    printf("Allocating %d bytes of memory.\n", imageCols*imageCols*imageCount);
	//The dataset is very larege, so use malloc to allocate memory to store it.
    uint8_t (*image)[imageRows][imageCols] = malloc(imageCount * sizeof(*image));

    //Load the data into memory.
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
    FILE * labelData = fopen(TRAINING_LABEL_PATH,"rb"); //Open training labels
    if (labelData == NULL)
    {
        printf("\nERROR OPENING LABEL DATA!!!\n"); //Check that file opened, exit on failure.
		return 0;
    }
    else
    {
        printf("Opened label data.\n");
    }

	//Declare unsigned 32 bit integer to hold information from the file.
    uint32_t label_magicNumber, labelCount;
    fread(&label_magicNumber, 4, 1, labelData); //Read the "magic number", which specifies the data type and how many dimensions
    fread(&labelCount, 4, 1, labelData);		//Read how many labels.

    label_magicNumber = __builtin_bswap32(label_magicNumber);
    labelCount = __builtin_bswap32(labelCount);

	//File format is big-endian. Reverse the bytes into little-endian so we can read them.
    printf("The label data contains:\n  %d labels\n", labelCount);
    printf("Loading labels into memory...\n");

	//Load labels into array
    uint8_t labels[labelCount];
    for(uint32_t labelIndex = 0; labelIndex < labelCount; labelIndex++)
    {
        fread(&labels[labelIndex], 1, 1, labelData);
    }

	//Choose to use blank B and R, or continue training from file
	if (oldDogNewTricks == true)
	{
		/* Load the B and R arrays from file */
		printf("Loading B and R from file...\n");
		read2DArrayFromFile(B,B_PARAMETER_PATH);
		read3DArrayFromFile(R,R_PARAMETER_PATH);
	}
	else if (resume == true)
	{
		/* Load the B and R arrays from file */
		printf("Loading backup of B and R from file...\n");
		read2DArrayFromFile(B,B_PARAMETER_BACKUP);
		read3DArrayFromFile(R,R_PARAMETER_BACKUP);

	}
	else
	{
		/* Seed the B and R arrays with random numbers */
		printf("Seeding B and R randomly...\n");
		seed3D(R);
		seed2D(B);
	}


	printf("---> Training started! <----");
	
	epochMissAverage = ERROR+1;
	while( cost > ERROR )
	// while( epochMissAverage > ERROR )
	{
		epochMiss=0;
		epochStart = clock();
		
		// Training loop
		for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
		{
			/*   Fill starting X with sample, Y with label.   */
			mu=0; //Start mu at zero
			//Loop through every cell in the current sample, loading the image into the zeroth layer of the network and the label into the desired output.
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
						Z[mu][t] += R[mu][nu][t]*X[nu][t-1];
					}
					X[mu][t] = sigma(Z[mu][t]);
				}
					
			}

			//Back propagate
			for (mu = 0; mu < N; mu++)
			{
				dB[mu][T] = -EPS*(X[mu][T]-Y[mu])*X[mu][T]*(1-X[mu][T]);
				B[mu][T] = B[mu][T] + dB[mu][T];
				for (nu = 0; nu < N; nu++)
				{
					R[mu][nu][T] += dB[mu][T]*X[nu][T-1];
				}
			}

			// k = 1...T-1 layers
			for (int k = 1; k < T; k++)
			{
				for (mu = 0; mu < N; mu++)
				{
					dB[mu][T-k] = 0;
					for (int a = 0; a < N; a++)
					{
						dB[mu][T-k] = dB[mu][T-k] + dB[a][T-k+1]*R[a][mu][T-k+1]*X[mu][T-k]*(1-X[mu][T-k]);
					}
					B[mu][T-k] = B[mu][T-k] + dB[mu][T-k];
					for (nu = 0; nu < N; nu++)
					{
						R[mu][nu][T-k] = R[mu][nu][T-k] + dB[mu][T-k]*X[nu][T-k-1];
					}
				}
			}

			//Calculate the average error over all output neurons
			float sumOfOutputLayer=0;
			for (int mu = 0; mu < N; mu++)
			{
				sumOfOutputLayer += X[mu][T]; //Add up all the neurons
			}
			output = sumOfOutputLayer/(double)N;
			miss = fabs(((double)labels[imageIndex]/10.0) - output);
			epochMiss += miss;

			// calculate cost function
			cost = 0;
			for (mu = 0; mu < N; mu++){
				cost = cost + (X[mu][T]-Y[mu])*(X[mu][T]-Y[mu]);
			}
			cost = 0.5*cost;

			if (imageIndex % 1000 == 0)
			{
				secondsSinceStart = timeSince(start);
				sprintf(formattedTime, "%02d:%02d:%02d", secondsSinceStart / 3600, (secondsSinceStart % 3600) / 60, secondsSinceStart % 60); 
                //printf("%d seconds since start is equal to %s (HH:MM:SS format)\n", secondsSinceStart, formattedTime);
				printf("\n%s | Sample %d -> Guessed %1.5f, answer %d, miss of %1.5f, cost: %1.5f", 
						formattedTime, imageIndex, output*10, labels[imageIndex], miss*10, cost );
			}
			// increment number of back propagations
			cycles++;

		} // End of loop over training data
		
		//Calculate average error across last epoch
		epochMissAverage=(epochMiss/(float)imageCount);
		printf("\n\n>>>>>>>>>>> EPOCH %ld MISS AVERAGE: %f ",epochs, epochMissAverage);
		printf("\nBacking up B and R...\n");
		write2DArrayToFile(B,B_PARAMETER_BACKUP);
		write3DArrayToFile(R,R_PARAMETER_BACKUP);
		
		//Log stats
	    char logBuffer[100];
        sprintf(logBuffer, "%f, %f\n", epochMissAverage, timeSince(start));
        //logToFile("learningCurve.log", logBuffer);

		epochs++;

	}//End training

	//Training is done. Now save the parameters to file.
	write2DArrayToFile(B,"B.neural");
	write3DArrayToFile(R,"R.neural");
	
	printf("#####################################################\n");
	printf("\n>>>>>>>>>>>>>>> TRAINING COMPLETE!!! <<<<<<<<<<<<<<<\n");
	printf("After %ld epochs the network reached %2.2f average miss.\n");
	printf("The results were saved to ./B.neural & ./R.neural ");
	printf("#####################################################\n");

}//End main

