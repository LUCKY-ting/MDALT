#include "mex.h"
#include "math.h"
#include "matrix.h"
#include "stdlib.h"
#include "float.h"
#include "time.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

double maxfun(double a, double b)
{
    if (a >= b) return a;
    else return b;
}

double minfun(double a, double b)
{
    if (a <= b) return a;
    else return b;
}

double uniform(double a, double b)
{
    return ((double) rand())/ RAND_MAX * (b -a) + a;
}

int binornd(double p)
{
    int x;
    double u;
    u = uniform(0.0, 1.0);
    x = (u <= p)? 1:0;
    return(x);
}


int getRandInt(int lowerLimit, int upperLimit){ // get an randomized interger in [lowerLimit, upperLimit]
    return lowerLimit + rand() % (upperLimit - lowerLimit + 1);
}

void randPerm(double *index, int N){
    int i, r1, r2, tmp;
    for(i=0; i < N; i++){
        r1 = getRandInt(0, N-1);
        r2 = getRandInt(0, N-1);
        if (r1!=r2){
            tmp =  index[r1];
            index[r1]= index[r2];
            index[r2] = tmp;
        }
    }
}

double squareNorm(double *x, int len){
    int i;
    double sum = 0;
    for(i = 0;i < len; i++){
        sum = sum + x[i] * x[i];
    }
    return sum;
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 8) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "8 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0 && mxIsSparse(prhs[1])==0){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "data/label matrix is not sparse!");
    }
    
    
    double *data, *labels, *w, *index, *x, *y, a_t, b_t, coeff, sum_coeff, delta, theta, eta, a_k_minus, a_k_plus, b_k_minus, b_k_plus;
    int i,j,k,p,N,d,L,low,high,nonzerosNum,low1,high1,epoch,o,Y_t_size,nY_t_size,iter,maxIterNum;
    mwIndex *ir, *jc, *ir1, *jc1;
    int *idx;
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    labels = mxGetPr(prhs[1]);
    index = mxGetPr(prhs[2]);
    delta = mxGetScalar(prhs[3]);
    theta = mxGetScalar(prhs[4]);
    eta = mxGetScalar(prhs[5]);
    maxIterNum = mxGetScalar(prhs[6]);
    epoch = mxGetScalar(prhs[7]);
    
    // a column is an instance
    d = (int)mxGetM(prhs[0]); //get Number of rows in array
    N = (int)mxGetN(prhs[0]); //get Number of columns in array
    L = (int)mxGetM(prhs[1]); //the dimension of each label vector
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    ir1 = mxGetIr(prhs[1]);
    jc1 = mxGetJc(prhs[1]);
    
    /* preparing outputs */
    plhs[0] = mxCreateDoubleMatrix(d, L+1, mxREAL);
    w = mxGetPr(plhs[0]);
    
    // plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    // errNum = mxGetPr(plhs[1]);
    
    double * pred_v = Malloc(double,L+1);
    srand(0);
    
    for (o = 1; o <= epoch; o++){
        if (o > 1) randPerm(index, N);
        /* start loop */
        for(i = 0; i < N; i++)
        {
            j = index[i] - 1;
            // get each instance
            low = jc[j]; high = jc[j+1];
            nonzerosNum = high - low;
            x = Malloc(double,nonzerosNum);
            idx = Malloc(int,nonzerosNum); // the indices of the non-zero values in x
            for (k = low; k < high; k++){
                x[k-low] = data[k];
                idx[k-low] = ir[k];
            }
            
            // get each label vector
            y = Malloc(double,L);
            for (k = 0; k < L; k++){
                y[k] = -1;
            }
            low1 = jc1[j]; high1 = jc1[j+1];
            for (k = low1; k < high1; k++){
                y[ir1[k]] = labels[k];
            }
            Y_t_size = high1 - low1; // the number of relevant labels
            nY_t_size = L - Y_t_size; // the number of irrelevant labels
           
            int *R_Idx = Malloc(int,Y_t_size);
            int *nR_Idx = Malloc(int,nY_t_size);
            int r = 0, s = 0;
            // compute each predicted value
            for (k = 0; k <= L; k++){
                pred_v[k] = 0;
                for (p = 0; p < nonzerosNum; p++){
                    pred_v[k] += w[k*d + idx[p]] * x[p];
                }
                if (k < L){ //compute the indices of relevant and irrelevant labels
                    if (y[k] == 1){
                        R_Idx[r++] = k;
                    }else
                        nR_Idx[s++] = k;
                }
            }
            double x_norm = squareNorm(x, nonzerosNum);
            for (iter = 0; iter < maxIterNum; iter++){
                double *gamma = Malloc(double, 2*L);
                double *pailie = Malloc(double, 2*L);
                for (k = 0; k < 2*L; k++){
                    gamma[k] = 0;
                    pailie[k] = k;
                }
                double func_value = 0;
                double ratio = 1e10;
                double part_sum1 = 0, part_sum2 = 0, part_sum3 = 0, part_sum4 = 0;
                while(fabs(ratio) > 1e-3){
                    double old_func_value = func_value;
                    double q_h, tmp, kappa_h, gradient,old_gamma, uncut;
                    randPerm(pailie, 2*L);
                    for (k = 0; k < 2*L; k++){  //updating the h-th component of gamma
                        int h = pailie[k];
                        if (h < Y_t_size){
                            q_h = pred_v[R_Idx[h]] - pred_v[L] - 1 + theta;
                            tmp = gamma[h] - gamma[h+Y_t_size] + part_sum1 + part_sum4 - part_sum2 - part_sum3;
                            kappa_h = eta/(1 - theta) * 1.0 /Y_t_size;
                        }else if (h < 2*Y_t_size){
                            q_h = -(pred_v[R_Idx[h-Y_t_size]] - pred_v[L] - 1) + theta;
                            tmp = gamma[h] -gamma[h-Y_t_size] + part_sum2 + part_sum3 - part_sum1 - part_sum4;
                            kappa_h = eta/(1 - theta) * delta/Y_t_size;
                        }else if (h < L+ Y_t_size){
                            q_h = pred_v[L] - pred_v[nR_Idx[h - 2*Y_t_size]] - 1 + theta;
                            tmp = gamma[h] - gamma[h+nY_t_size] + part_sum2 + part_sum3 - part_sum1 - part_sum4;
                            kappa_h = eta/(1 - theta) * 1.0 /nY_t_size;
                        }else{
                            q_h = -(pred_v[L] - pred_v[nR_Idx[h-L-Y_t_size]] - 1) + theta;
                            tmp = gamma[h] - gamma[h-nY_t_size] + part_sum1 + part_sum4 - part_sum2 - part_sum3;
                            kappa_h = eta/(1 - theta) * delta /nY_t_size;
                        } 
                        gradient = x_norm * tmp + q_h;
                        if (gradient!=0){
                            old_gamma = gamma[h];
                            uncut = gamma[h] - gradient / (2 * x_norm);
                            gamma[h] = minfun(maxfun(0, uncut), kappa_h);
                            //update 4 part_sum 
                            if (h < Y_t_size){
                                part_sum1 = part_sum1 - old_gamma + gamma[h];
                            }else if (h < 2*Y_t_size){
                                part_sum2 = part_sum2 - old_gamma + gamma[h];
                            }else if (h < L+ Y_t_size){
                                part_sum3 = part_sum3 - old_gamma + gamma[h];
                            }else
                                part_sum4 = part_sum4 - old_gamma + gamma[h];
                        }
                    }
                    
                    // compute the objective function value
                    double part1 = 0, part3 = 0, part4 = 0, part6 = 0;
                    for (p = 0; p < Y_t_size; p++){
                        part1 = part1 + (gamma[p] - gamma[p+Y_t_size])*(gamma[p] - gamma[p+Y_t_size]);
                        part3 = part3 + (gamma[p] - gamma[p+Y_t_size]);
                        part4 = part4 + (gamma[p] - gamma[p+Y_t_size])* (pred_v[R_Idx[p]] - pred_v[L] - 1);
                        part6 = part6 + (gamma[p] + gamma[p+Y_t_size]);
                    }
                    double part2 = 0, part5 = 0;
                    for (p = 0; p < nY_t_size; p++){
                        part2 = part2 + (gamma[p+2*Y_t_size] - gamma[p+L+Y_t_size])*(gamma[p+2*Y_t_size] - gamma[p+L+Y_t_size]);
                        part3 = part3 - (gamma[p+2*Y_t_size] - gamma[p+L+Y_t_size]);
                        part5 = part5 + (gamma[p+2*Y_t_size] - gamma[p+L+Y_t_size]) * (pred_v[L] - pred_v[nR_Idx[p]] - 1);
                        part6 = part6 + (gamma[p+2*Y_t_size] + gamma[p+L+Y_t_size]);
                    }
                    func_value = (part1 + part2 + part3*part3)*x_norm / 2 + part4 + part5 + theta * part6;
                    if (func_value - old_func_value == 0)
                        ratio = 0;
                    else if (old_func_value == 0)
                         ratio = 1e10;
                    else
                        ratio = (func_value - old_func_value)/old_func_value;
                    //printf("func_value = %f, ratio = %f \n", func_value, ratio);
                }

                //update the scoring model and its predictive value
                int releNum = 0, nreleNum = 0;
                for (k = 0; k < L; k++){
                    if (y[k] == 1){
                        pred_v[k] = 0;
                        for (p = 0; p < nonzerosNum; p++){
                            w[k*d + idx[p]] = w[k*d + idx[p]] + (gamma[releNum] - gamma[releNum+Y_t_size]) * x[p];
                            w[L*d + idx[p]] = w[L*d + idx[p]] - (gamma[releNum] - gamma[releNum+Y_t_size]) * x[p];
                            pred_v[k] += w[k*d + idx[p]] * x[p];
                        }
                        releNum = releNum + 1;
                    }else{
                        pred_v[k] = 0;
                        for (p = 0; p < nonzerosNum; p++){
                            w[k*d + idx[p]] = w[k*d + idx[p]] - (gamma[nreleNum+2*Y_t_size] - gamma[nreleNum+L+Y_t_size]) * x[p];
                            w[L*d + idx[p]] = w[L*d + idx[p]] + (gamma[nreleNum+2*Y_t_size] - gamma[nreleNum+L+Y_t_size]) * x[p];
                            pred_v[k] += w[k*d + idx[p]] * x[p];
                        }
                        nreleNum = nreleNum + 1;
                    }  
                }
                //update the predictive value of the thresholding model
                pred_v[L] = 0;
                for (p = 0; p < nonzerosNum; p++){
                    pred_v[L] += w[L*d + idx[p]] * x[p];
                }
                free(gamma);
                free(pailie);
            }
            free(x);
            free(idx);
            free(y);
            free(R_Idx);
            free(nR_Idx);
        }
    }
    free(pred_v);
}

