#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>



const float EPS 0.2			//Learning rate
const float ERROR 0.0001	//Target ERROR






//Sigma function
double sigma(double input)
{
	return (1/(1+exp(-input)));
}


seedRandom()
{
	srand(time(NULL));


}




