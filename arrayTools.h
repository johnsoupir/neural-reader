#ifndef ARRAY_TOOLS_H
#define ARRAY_TOOLS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

 #define N 784
 #define T 3

void write3DArrayToFile(double array[N][N][T+1], const char* filename);
void write2DArrayToFile(double array[N][T+1], const char* filename);
void read3DArrayFromFile(double array[N][N][T+1], const char* filename);
void read2DArrayFromFile(double array[N][T+1], const char* filename);
void print3DArray(double array[N][N][T+1]);
void print2DArray(double array[N][T+1]);
void seed3D(double array[N][N][T+1]);
void seed2D(double array[N][T+1]);

#endif // ARRAY_TOOLS_H

