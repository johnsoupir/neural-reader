#include <stdio.h>
#include <math.h>

#define N 2 // num neurons per layer
#define T 2 // num hidden layers
#define EPS 0.5 // learning rate
#define EPOCHS 16000 // number of back propagations

double sigma(double x);
int main (void){
	double X[N][T+1], Z[N][T+1];
	double R[N][N][T+1], B[N][T+1];
	double Y[N], A[N][T+1];
	int mu, nu, beta, t;

	R[0][0][0] = 0.1;
	R[0][1][0] = 0.2;
	R[1][0][0] = 0.3;
	R[1][1][0] = 0.4;
	R[0][0][1] = 0.5;
	R[0][1][1] = 0.6;
	R[1][0][1] = 0.7;
	R[1][1][1] = 0.8;
	B[0][1] = 0.1;
	B[1][1] = 0.2;
	B[0][2] = 0.3;
	B[1][2] = 0.4;
	X[0][0] = 10;
	X[1][0] = 10;
	Y[0] = 1;
	Y[1] = 0.5;

for(int epoch = 0; epoch < EPOCHS; epoch++){
	// forward propagate
	for (t = 0; t < T; t++){
		for (mu = 0; mu < N; mu++){
			Z[mu][t+1] = B[mu][t+1];
			for (nu = 0; nu < N; nu++){
				Z[mu][t+1] = Z[mu][t+1] + R[mu][nu][t]*X[nu][t];
			}
			X[mu][t+1] = sigma(Z[mu][t+1]);
		}
	}
	// back propagate
	// output layer first
	for (mu = 0; mu < N; mu++){
		A[mu][T-1] = (X[mu][T]-Y[mu])*X[mu][T]*(1-X[mu][T]);
		B[mu][T-1] = B[mu][T-1]-EPS*A[mu][T-1];
		for (nu = 0; nu < N; nu++){
			R[mu][nu][T-1] = R[mu][nu][T-1]-EPS*A[mu][T-1]*X[nu][T-1];
		}
	}
	// back through remaining layers
	for (t = T-1; t > 0; t--){
		for (mu = 0; mu < N; mu++){
			A[mu][t-1] = 0;
			for (beta = 0; beta < N; beta++){
				A[mu][t-1] = A[mu][t-1] + A[beta][t]*R[beta][mu][t]*X[mu][t]*(1-X[mu][t]);
			}
			B[mu][t-1] = B[mu][t-1]-EPS*A[mu][t-1];
			for (nu = 0; nu < N; nu++){
				R[mu][nu][t-1] = R[mu][nu][t-1]-EPS*A[mu][t-1]*X[nu][t-1];
			}
		}
	} 
}
	// print out final values of R and B
	fprintf(stderr,"After %d back propagations the neural net\n",EPOCHS); 
	fprintf(stderr,"is now trained with the following parameters:\n"); 
	for (t = 0; t < T; t++){
		fprintf(stderr,"\nB[mu](%d)\tR[mu][nu](%d)\n", t, t); 
		for (mu = 0; mu < N; mu++){
			fprintf(stderr,"\n|%0.4lf|\t|", B[mu][t]); 
			for (nu = 0; nu < N; nu++){
				fprintf(stderr," %0.4lf", R[mu][nu][t]); 
			}
			fprintf(stderr," |"); 
		}
		fprintf(stderr,"\n"); 
	}
	fprintf(stderr,"\nFinal output and desired values are:\n\n"); 
	for (mu = 0; mu < N; mu++){
		fprintf(stderr,"X[%d][%d] = %0.4lf\t Y[%d] = %0.4lf\n", mu, T, X[mu][T], mu, Y[mu]); 
	}
	fprintf(stderr,"\n"); 
	return 0;
}
double sigma(double x){
	return 1/(1+exp(-x));
}
