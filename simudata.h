#ifndef CIRCLE_H
#define CIRCLE_H

#include <bits/stdc++.h>
#include "mkl.h"

#define MaxPheNum 1000

using namespace std;

namespace gsl {
    #include "gsl/gsl_linalg.h"
    #include "gsl/gsl_cdf.h"
    #include "gsl/gsl_randist.h"
}

int inputParam(string inputFileDir, 
               float *orgcorm, float *sigeffcorm, float *corm2, float *es, 
               int orgcormDim, int corm2Dim);
void mkdf(gsl::gsl_vector * generatorRlt, int snpn, gsl::gsl_rng *r, gsl::gsl_vector *zeroVector, gsl::gsl_matrix *A, float *tm, int cormDim);
int metafSimulation(float* tm, float*  bgc, float* efd, float thv, float* es, int dim, double &pm, int snpn);
int calcInverseMatrix(float* pDst, const float* pSrc, int dim);
int trtvf(vector<float>& v, vector<int>& extractId, int n, float* cm, int dim, double &rlt);
template <typename T>
void sort_indexes_abs_decrease(const vector<T> &v, vector<int> &idx);
int sortrt(float* tm, float* corm, int dim, float thv, double &pm, int snpn);
void timeElapse(timeval &time_start, string stepName);

#endif