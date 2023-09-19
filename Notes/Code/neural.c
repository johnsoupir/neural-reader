#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 2           // num neurons per layer
#define EPS 0.5       // learning rate
#define ERROR 0.00001 // end when error reaches this

double sigma(double x){return 1/(1+exp(-x));}

int main (int argc, char **argv){
	if (argc != 2*N+2){
		fprintf(stderr,"need %d inputs, %d correct answers\n", N, N); 
		fprintf(stderr,"followed by the number of hidden layers\n"); 
		fprintf(stderr,"EX: %s 0.05 0.10 0.01 0.99 2\n", argv[0]); 
		return 1;
	}
	srand(time(NULL));
	int T = atoi(argv[argc-1]);
	double Y[N], cost=ERROR+1;
	double X[N][T+1],Z[N][T+1],dB[N][T+1];
	double R[N][N][T+1],B[N][T+1];
	int mu, nu, t=0, cycles=0;
	for (mu = 0; mu < N; mu++){
		X[mu][0] = atof(argv[mu+1]);
		Y[mu] = atof(argv[mu+N+1]);
	}
	// randomly initialize parameters
	for (t = 1; t <= T; t++){
		for (mu = 0; mu < N; mu++){ 
			B[mu][t] = (double) (rand()%1000)/1000.0;
			for (nu = 0; nu < N; nu++){
				R[mu][nu][t] = (double) (rand()%1000)/1000.0;
			}
		}
	}
	while( cost > ERROR ){
		// forward propagate
		for (t = 1; t <= T; t++){
			for (mu = 0; mu < N; mu++){
				Z[mu][t] = B[mu][t];
				for (nu = 0; nu < N; nu++){
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
		// calculate cost function
		cost = 0;
		for (mu = 0; mu < N; mu++){
			cost = cost + (X[mu][T]-Y[mu])*(X[mu][T]-Y[mu]);
		}
		cost = 0.5*cost;
		// increment number of back propagations
		cycles++;
	}
	// print out final values of R and B
	fprintf(stdout,"\nThe trained network parameters:\n"); 
	for (t = 1; t <= T; t++){
		fprintf(stdout,"\nB[mu](%d)\t  R[mu][nu](%d,%d)\n", t, t, t-1); 
		for (mu = 0; mu < N; mu++){
			fprintf(stdout,"\n|%6.3lf |\t|", B[mu][t]); 
			for (nu = 0; nu < N; nu++){
				fprintf(stdout," %6.3lf ", R[mu][nu][t]); 
			}
			fprintf(stdout," |"); 
		}
		fprintf(stdout,"\n"); 
	}
	fprintf(stdout,"After %d back propagations:\n\n",cycles); 
	// print out final values X
	fprintf(stdout,"OUTPUT\t\t\tCORRECT OUTPUT\n"); 
	fprintf(stdout,"------\t\t\t--------------\n"); 
	for (mu = 0; mu < N; mu++){
		fprintf(stdout,"X[%d][%d] = %0.2lf \t\tY[%d] = %0.2lf\n", mu, T, X[mu][T], mu, Y[mu]); 
	}
	return 0;
}
