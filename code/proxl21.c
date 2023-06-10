/*=================================================================
 *
 * proxl21.c, proxl21.mex: 
 *
 * projection onto l21 norm
 *
 * b = proxl21(A,rw)
 *
 * This is a MEX-file for MATLAB.  
 *
 *=================================================================*/


#include "mex.h" /* Always include this */
#include <math.h>

/* Input Arguments */
#define A_IN prhs[0]
#define P_IN prhs[1]

/* Output Arguments */
#define B_OUT plhs[0]

void DoComputation(double *B, double *A, int M, int N, double p)
{
    double colnorm;
    int m, n;
    for(n = 0; n < N; n++)
    {
    /* Compute the norm of the nth column */
        for(m = 0, colnorm = 0.0; m < M; m++)
            colnorm += pow(A[m + M*n], 2);
        colnorm = pow(fabs(colnorm), 1.0/2);
        /* Fill the nth column of B */
        if(colnorm>p){
            double absnorm= (colnorm-p)/colnorm;
            for(m = 0; m < M; m++)
                B[m + M*n] =absnorm*A[m + M*n];
        }else{
            for(m = 0; m < M; m++)
                B[m + M*n] = 0;
    }
    }
}


void mexFunction(int nlhs, mxArray *plhs[], /* Outputs */
    int nrhs, const mxArray *prhs[]) /* Inputs */
{
    double p;
    /*** Check inputs ***/
    if(nrhs < 1 || nrhs > 2)
        mexErrMsgTxt("Must have either 1 or 2 input arguments.");
    if(nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");
    if(mxIsComplex(A_IN) || mxGetNumberOfDimensions(A_IN) != 2 || mxIsSparse(A_IN) || !mxIsDouble(A_IN))
        mexErrMsgTxt("Sorry! A must be a real 2D full double matrix.");
    if(nrhs == 1) /* Was p not specified? */
        p = 2.0; /* Set default value for p */
    else
    if(mxIsComplex(P_IN) || !mxIsDouble(P_IN) || mxGetNumberOfElements(P_IN) != 1)
        mexErrMsgTxt("p must be a double scalar.");
    else
    p = mxGetScalar(P_IN); /* Get the value of p */
    
    int M, N;
    double *A, *B;
    
    M = mxGetM(A_IN); /* Get the dimensions of A */
    N = mxGetN(A_IN);
    A = mxGetPr(A_IN); /* Get pointer to A's data */
    /*** Create the output matrix ***/
    B_OUT = mxCreateDoubleMatrix(M, N, mxREAL);
    B = mxGetPr(B_OUT);
    DoComputation(B, A, M, N, p);
}