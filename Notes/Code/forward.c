#include <stdio.h>
#include <math.h>

#define N 2 // num neurons
#define T 2 // num hidden layers

double sigma(double x);
int main (void){
	double X[N][T+1],Z[N][T+1];
	double R[N][N][T+1],B[N][T+1];
	int mu, nu;

	R[0][0][0] = 1;
	R[0][1][0] = 2;
	R[1][0][0] = 3;
	R[1][1][0] = 4;
	R[0][0][1] = 5;
	R[0][1][1] = 6;
	R[1][0][1] = 7;
	R[1][1][1] = 8;
	B[0][1] = 1;
	B[1][1] = 2;
	B[0][2] = 3;
	B[1][2] = 4;
	X[0][0] = 10;
	X[1][0] = 10;

	// forward propagate

	for (int t = 0; t < T; t++){
		for (mu = 0; mu < N; mu++){
			Z[mu][t+1] = B[mu][t+1];
			for (nu = 0; nu < N; nu++){
				Z[mu][t+1] = Z[mu][t+1] + R[mu][nu][t]*X[nu][t];
			}
			X[mu][t+1] = sigma(Z[mu][t+1]);
			fprintf(stderr,"X[%d][%d] = %lf\n", mu, t+1, X[mu][t+1]); 
		}
	}
	return 0;
}
double sigma(double x){
	return 1/(1+exp(-x));
}
