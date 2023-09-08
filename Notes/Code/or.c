#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPS 0.2       // learning rate
#define ERROR 0.00001 // end when error reaches this

double sigma(double x){return 1/(1+exp(-x));}

int main (int argc, char **argv){
	if (argc != 3){
		fprintf(stderr,"need the number of neurons per layer\n"); 
		fprintf(stderr,"and the number of hidden layers\n"); 
		fprintf(stderr,"EX: %s 4 2\n", argv[0]); 
		return 1;
	}
	srand(time(NULL));
	int N = atoi(argv[1]);
	int T = atoi(argv[2]);
	int mu, nu, t=0, cycles=0;
	double Y[N],cost=ERROR+1;
	double R[N][N][T+1],B[N][T+1];
	double X[N][T+1],Z[N][T+1],dB[N][T+1];
	double IN[4][N],OUT[4][N];

	// initialize all inputs and outputs
	for(int i = 0; i < 4; i++){
		for (mu = 0; mu < N; mu++){
			IN[i][N] = 0.001;
			OUT[i][N] = 0.001;
		}
	}
	// inputs and desired outputs for OR gate
	// Note that we only care about the first 
	// two neurons, all others remain zero
  // For the output, all 1's is a 1, all 0's
  // is a 0
	IN[0][0] = 1;
	IN[0][1] = 1;
	OUT[0][0] = 1;
	IN[1][0] = 1;
	IN[1][1] = 0;
	OUT[1][0] = 1;
	IN[2][0] = 0; 
	IN[2][1] = 1;
	OUT[2][0] = 1;
	IN[3][0] = 0; 
	IN[3][1] = 0;
	OUT[3][0] = 0;

	// randomly initialize network parameters
	for (t = 1; t <= T; t++){
		for (mu = 0; mu < N; mu++){
			B[mu][t] = (double) (rand()%1000)/1000.0;
			for (nu = 0; nu < N; nu++){
				R[mu][nu][t] = (double) (rand()%1000)/1000.0;
			}
		}
	}

	while( cost > ERROR ){
		// loop over training data
		for(int i = 0; i < 4; i++){
			for (mu = 0; mu < N; mu++){
				X[mu][0] = IN[i][mu];
				Y[mu] = OUT[i][0];
			}
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
		} // end training data loop
	} // end cost while loop

	// Now the network should be trained.
	// print out final values of R and B
	fprintf(stdout,"\nThe trained network parameters\n"); 
	fprintf(stdout,"after %d back propagations:\n", cycles); 
	for (t = 1; t <= T; t++){
		fprintf(stdout,"\nB[mu](%d)\t  R[mu][nu](%d,%d)\n", t, t, t-1); 
		for (mu = 0; mu < N; mu++){
			fprintf(stdout,"\n|%6.3lf |\t|", B[mu][t]); 
			for (nu = 0; nu < N; nu++){
				fprintf(stdout," %6.3lf", R[mu][nu][t]); 
			} fprintf(stdout," |"); 
		} fprintf(stdout,"\n"); 
	}

	// Now test the trained network
	fprintf(stdout,"\nTesting the trained network...\n"); 
	for(int i = 0; i < 4; i++){
		for (mu = 0; mu < N; mu++){
			X[mu][0] = IN[i][mu];
			Y[mu] = OUT[i][0];
		}
		for (t = 1; t <= T; t++){
			for (mu = 0; mu < N; mu++){
				Z[mu][t] = B[mu][t];
				for (nu = 0; nu < N; nu++){
					Z[mu][t] = Z[mu][t] + R[mu][nu][t]*X[nu][t-1];
				}
				X[mu][t] = sigma(Z[mu][t]);
			}
		}
		// print out results from this input
		fprintf(stdout,"\nINPUT\t\t\t\t\t\tOUTPUT\n"); 
		fprintf(stdout,"-----\t\t\t\t\t\t------\n"); 
		for (mu = 0; mu < N; mu++){
			fprintf(stdout,"X[%d][0] = %0.2lf\tX[%d][%d] = %0.2lf\n", mu, X[mu][0], mu, T, X[mu][T]); 
		}
	}
	return 0;
}
