#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


double sigma(double x){return 1/(1+exp(-x));}

int main (int argc, char **argv){

	if (argc != 1){
		fprintf(stderr,"\n No arguments required.\n"); 
		fprintf(stderr,"\n Place B and R file is this directory..\n"); 
	}


	int N = 2; //atoi(argv[1]);
	int T = 4; //atoi(argv[2]);

	int mu, nu, t=0;

	double Y[N]; 					//Answer layer - holds what the expected output is
	double R[N][N][T+1],B[N][T+1];				//Transition and bias
	double X[N][T+1],Z[N][T+1];		//
	double IN[4][N],OUT[4][N];					//In = samples, out = correct answer

	//Load B and R from file
	FILE * bFile = fopen("B.neural","rb");
	fread(B, sizeof(double), N*(T+1), bFile);
	fclose(bFile);

	FILE * rFile = fopen("R.neural","rb");
	fread(R, sizeof(double), N*N*(T+1), rFile);
	fclose(rFile);


	// initialize all inputs and outputs - fills as many neurons as you want with small values
	for(int i = 0; i < 4; i++){
		for (mu = 0; mu < N; mu++){
			IN[i][N] = 0.001;
			OUT[i][N] = 0.001;
		}
	}
	

	//Inputs and expected outputs
	IN[0][0] = 1;
	IN[0][1] = 1;
	OUT[0][0] = 0;

	IN[1][0] = 1;
	IN[1][1] = 0;
	OUT[1][0] = 1;
	
	IN[2][0] = 0; 
	IN[2][1] = 1;
	OUT[2][0] = 1;

	IN[3][0] = 0; 
	IN[3][1] = 0;
	OUT[3][0] = 1;

	// loop over training data
	for(int i = 0; i < 4; i++){

		//Load samples and answers
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
		
		// print out results from this input
		fprintf(stdout,"\nINPUT\t\t\t\t\t\tOUTPUT\n"); 
		fprintf(stdout,"-----\t\t\t\t\t\t------\n"); 
		for (mu = 0; mu < N; mu++){
			fprintf(stdout,"X[%d][0] = %0.2lf\t\t\t\t\tX[%d][%d] = %0.2lf\n", mu, X[mu][0], mu, T, X[mu][T]); 
		}
	} // end training data loop

	
	return 0;
}
