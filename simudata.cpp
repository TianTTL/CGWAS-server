#include "simudata.h"
#include <bits/stdc++.h>
#include <sys/time.h>
#include "mkl.h"
#include <omp.h>

using namespace std;

namespace gsl {
    #include "gsl/gsl_linalg.h"
    #include "gsl/gsl_cdf.h"
    #include "gsl/gsl_randist.h"
}

int main(int argc, char * argv[]) {
    int status; // error handle
    int swit, orgcormDim, corm2Dim, snpn, simulateTimes;
    float thv;
    float *orgcorm = new float [MaxPheNum * MaxPheNum];
    float *corm2 = new float [MaxPheNum * MaxPheNum];
    float *es = new float [MaxPheNum];
    float *sigeffcorm = new float [MaxPheNum * MaxPheNum * 2];
    int threadsNum;
    struct timeval time_start;

    // time stamp
    gettimeofday(&time_start,NULL);

    // input data
    string inputFileDir(argv[1]);
    string outputFileDir(argv[2]);
    swit = atoi(argv[3]);
    snpn = atoi(argv[4]);
    thv = atof(argv[5]);
    orgcormDim = atoi(argv[6]);
    corm2Dim = atoi(argv[7]);
    simulateTimes= atoi(argv[8]);
    threadsNum = atoi(argv[9]);
    status = inputParam(inputFileDir, orgcorm, sigeffcorm, corm2, es, orgcormDim, corm2Dim);
    if (status != 0) {
        cout << "input file error!\n";
        return(1);
    }

    timeElapse(time_start, "input");

    // pretreatment
    static float* tm;
    static gsl::gsl_vector * rngGMVOrgcorm;
    static gsl::gsl_vector * rngGMVcorm2;
#   pragma omp threadprivate(tm)
#   pragma omp parallel num_threads(threadsNum) \
    shared(snpn, orgcormDim, corm2Dim)
    {
       tm = new float [snpn * max(orgcormDim, corm2Dim)];
    }
    double pm1 = 0, pm2 = 0, pm3 = 0;
    double* simup = new double[3 * simulateTimes];
    MKL_INT errcode;

    // prepare for mkdf
    // initialize the array to 0
    float * zeroVector_orgcorm = new float[orgcormDim]();
    float * zeroVector_corm2 = new float[corm2Dim]();
    // rng seed
    VSLStreamStatePtr stream;
    VSLStreamStatePtr streamPriv[threadsNum];
    MKL_INT BRNG = VSL_BRNG_MCG31;
    MKL_INT SEED = (unsigned)time(0);
    errcode = vslNewStream(&stream, BRNG, SEED); // Creat main stream
    for(int i = 0; i < threadsNum; i++)
    {
        errcode = vslCopyStream(&streamPriv[i], stream);
        errcode = vslLeapfrogStream(streamPriv[i], (MKL_INT)i, threadsNum);
    }
    //cholesky for corm
    float* T_orgcorm =  new float [orgcormDim * orgcormDim];
    for (int i = 0; i < orgcormDim * orgcormDim; i++) { T_orgcorm[i] = orgcorm[i]; }
    float* T_corm2 =  new float [corm2Dim * corm2Dim];
    for (int i = 0; i < corm2Dim * corm2Dim; i++) { T_corm2[i] = corm2[i]; }
    errcode = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'U', (lapack_int)orgcormDim, T_orgcorm, (lapack_int)orgcormDim);
    errcode = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'U', (lapack_int)corm2Dim, T_corm2, (lapack_int)corm2Dim);

    timeElapse(time_start, "pretreatment");

    // simudata main step
#   pragma omp parallel for \
    num_threads(threadsNum) \
    private(pm1, pm2, pm3) \
    shared(swit, snpn, thv, orgcorm, corm2, orgcormDim, corm2Dim, streamPriv, zeroVector_orgcorm, zeroVector_corm2, T_orgcorm, T_corm2, simup) \
    schedule(dynamic)
    for (int i = 0; i < simulateTimes; i++) {
        int threadID = omp_get_thread_num();
        errcode = vsRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2, 
                    streamPriv[threadID], snpn, tm, 
                    orgcormDim, VSL_MATRIX_STORAGE_FULL, zeroVector_orgcorm, T_orgcorm);
        metafSimulation(tm, orgcorm, sigeffcorm, thv, es, orgcormDim, pm1, snpn);
        sortrt(tm, orgcorm, orgcormDim, thv, pm3, snpn);
        if (swit) {
            errcode = vsRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2, 
                                streamPriv[threadID], snpn, tm, 
                                corm2Dim, VSL_MATRIX_STORAGE_FULL, zeroVector_corm2, T_corm2);
            sortrt(tm, corm2, corm2Dim, thv, pm2, snpn);
        }

        simup[i * 3] = pm1;
        simup[i * 3 + 1] = pm2; // if swit is false, then pm2 == 0
        simup[i * 3 + 2] = pm3;
    }

    timeElapse(time_start, "main");

    // output simup
    string outputFilePath;
    fstream outputFile;
    outputFilePath = outputFileDir + "/simup";
    outputFile.open(outputFilePath.data(), ios::out);
    outputFile << scientific << setprecision(5);
    for (int i = 0; i < simulateTimes; i++) {
        outputFile << simup[i * 3] << "\t";
        if (swit) { outputFile << simup[i * 3 + 1] << "\t"; }
        outputFile << simup[i * 3 + 2] << "\n";
    }
    outputFile.close();

    // release memory
    #pragma omp parallel num_threads(threadsNum)
    {
       delete[] tm;
    }
    delete[] zeroVector_orgcorm;
    delete[] zeroVector_corm2;
    delete[] orgcorm;
    delete[] corm2;
    delete[] sigeffcorm;
    delete[] es;

    timeElapse(time_start, "output & free memory");

    return(0);
}

int inputParam(string inputFileDir, 
               float *orgcorm, float *sigeffcorm, float *corm2, float *es, 
               int orgcormDim, int corm2Dim) {
    string inputFilePath;
    fstream inputFile;

    // orgcorm
    inputFilePath = inputFileDir + "/orgcorm";
    inputFile.open(inputFilePath.data(), ios::in);
    for (int i = 0; i < orgcormDim; i++) {
        for (int j = 0; j < orgcormDim; j++) {
            inputFile >> orgcorm[i * orgcormDim + j];
        }
    }
    inputFile.close();

    // sigeffcorm
    inputFilePath = inputFileDir + "/sigeffcorm";
    inputFile.open(inputFilePath.data(), ios::in);
    for (int i = 0; i < orgcormDim; i++) {
        for (int j = 0; j < orgcormDim * 2; j++) {
            inputFile >> sigeffcorm[i * orgcormDim * 2 + j];
        }
    }
    inputFile.close();

    // corm2
    inputFilePath = inputFileDir + "/corm2";
    inputFile.open(inputFilePath.data(), ios::in);
    for (int i = 0; i < corm2Dim; i++) {
        for (int j = 0; j < corm2Dim; j++) {
            inputFile >> corm2[i * corm2Dim + j];
        }
    }
    inputFile.close();

    // es
    inputFilePath = inputFileDir + "/es";
    inputFile.open(inputFilePath.data(), ios::in);
    for (int i = 0; i < orgcormDim; i++) {
        inputFile >> es[i];
    }
    inputFile.close();
}

int metafSimulation(float* tm, float*  bgc, float* efd, float thv, float* es, int dim, double &pm, int snpn) {
    float* scm = new float [dim * dim];
    float* sigwv = new float [dim * dim * 2];
    float* sigwvByScm = new float [dim * dim * 2];
    float* sigwvByScmDotSigwvRowsumSqrt = new float [dim * 2];
    float* coefm = new float [dim * dim * 2];
    float* rlt = new float [dim * 2];
    vector <float> bgcExtract;
    vector <float> efdExtract;
    vector <float> esExtract;
    vector <float> tvExtract;
    vector <int> extractId;
    MKL_INT extractLen;
    int status;
    float maxChisq;
    MKL_INT incx = 1, incy = 1;
    float alpha = 1.0f, beta = 0.0f;

    // init pm
    pm = 1.0;
    
    for (int i = 0; i < snpn; i++) { // extract each line of tm
        extractId.clear();
        for (int j = 0; j < dim; j++) {
            if (abs(tm[i * dim + j]) > thv) {
                extractId.push_back(j);
            }
        }
        extractLen = extractId.size();
        if (extractLen == 0) {
            pm = min(pm, 1.0);
            continue;
        }

        bgcExtract.clear(); efdExtract.clear(); esExtract.clear(); tvExtract.clear();
        for (auto j : extractId) {
            for (auto k : extractId) {
                bgcExtract.push_back(bgc[j * dim + k]);
                efdExtract.push_back(efd[j * dim * 2 + k]);
            }
            for (auto k : extractId) {
                efdExtract.push_back(efd[j * dim * 2 + k + dim]);
            }
            esExtract.push_back(es[j]);
            tvExtract.push_back(tm[i * dim + j]);
        }

        status = calcInverseMatrix(scm, bgcExtract.data(), extractLen);
        if (status != 0) {
            pm = min(pm, 1.0);
            continue;
        }

        // calculate and transpose sigwv
        for (int j = 0; j < extractLen; j++) {
            for (int k = 0; k < 2 * extractLen;k++) {
                sigwv[k * extractLen + j] = efdExtract[j * extractLen * 2 + k] * esExtract[j];
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    2 * extractLen, extractLen, extractLen, 
                    alpha, sigwv, extractLen, scm, extractLen, 
                    beta, sigwvByScm, extractLen);
        for (int j = 0; j < 2 * extractLen; j++) {
            sigwvByScmDotSigwvRowsumSqrt[j] = 0;
            for (int k = 0; k < extractLen;k++) {
                sigwvByScmDotSigwvRowsumSqrt[j] += sigwvByScm[j * extractLen + k] * sigwv[j * extractLen + k];
            }
            sigwvByScmDotSigwvRowsumSqrt[j] = sqrt(sigwvByScmDotSigwvRowsumSqrt[j]);
        }

        // calc coefm
        for (int j = 0; j < 2 * extractLen; j++) {
            for (int k = 0; k < extractLen;k++) {
                coefm[j * extractLen + k] = sigwvByScm[j * extractLen + k] / sigwvByScmDotSigwvRowsumSqrt[j];
            }
        }

        cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                    2 * extractLen, extractLen, 
                    alpha, coefm, extractLen, 
                    tvExtract.data(), incx, 
                    beta, rlt, incy);
        maxChisq = 0;
        for (int j = 0; j < 2 * extractLen; j++) {
            if (abs(rlt[j]) > maxChisq) {
                maxChisq = abs(rlt[j]);
            }
        }
        pm = min(pm, gsl::gsl_cdf_chisq_Q(maxChisq * maxChisq, 1));
    }

    // free memory
    delete[] scm;
    delete[] sigwv;
    delete[] sigwvByScm;
    delete[] sigwvByScmDotSigwvRowsumSqrt;
    delete[] coefm;
    delete[] rlt;

    return(0);
}

int calcInverseMatrix(float* pDst, const float* pSrc, int dim) {
    int nRetVal = 0;

    int* ipiv = new int[dim];
    float* pSrcBak = new float[dim * dim];

    memcpy(pSrcBak, pSrc, sizeof(float)* dim * dim);
    memset(pDst, 0.f, sizeof(float)* dim * dim);
    for (int i = 0; i < dim; ++i) {
        pDst[i*dim + i] = 1.0;
    }

    MKL_INT N = dim;
    nRetVal = LAPACKE_sgesv(LAPACK_ROW_MAJOR, N, N, pSrcBak, N, ipiv, pDst, N);

    // free memory
    delete[] ipiv;
    delete[] pSrcBak;
    return nRetVal;
}

int trtvf(vector<float>& tv, vector<int>& extractId, int n, float* corm, int dim, double &rlt) {
    vector <float> cmExtract;
    vector <float> vExtract;
    float *cmSolve = new float [n * n];
    float *cmSolveByV = new float [n];
    int status;
    MKL_INT N = static_cast<MKL_INT> (n);
    float alpha = 1.0;
    float beta = 0.0;
    MKL_INT incx = 1, incy = 1;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cmExtract.push_back(corm[extractId[i] * dim + extractId[j]]);
        }
        vExtract.push_back(tv[extractId[i]]);
    }
    status = calcInverseMatrix(cmSolve, cmExtract.data(), n);
    if (status != 0) {
        return(1);
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                N, N, alpha, cmSolve, N, 
                vExtract.data(), incx, 
                beta, cmSolveByV, incy);
    float sumChisq = 0;
    for (int i = 0; i < n; i++) {
        sumChisq += cmSolveByV[i] * vExtract[i];
    }
    rlt = gsl::gsl_cdf_chisq_Q(sumChisq, n);

    // free memory
    delete[] cmSolve;
    delete[] cmSolveByV;

    return(0);
}

template <typename T>
void sort_indexes_abs_decrease(const vector<T> &v, vector<int> &idx) {
    // initialize original index locations
    // vector<size_t> idx(v.size());
    // iota(idx.begin(), idx.end(), 0);

    // get absolute value of each item in the v, 
    // and sort them in decreasing order
    stable_sort(idx.begin(), idx.end(), 
        [&v](size_t i1, size_t i2) {return abs(v[i1]) > abs(v[i2]);});
    return;
}

int sortrt(float* tm, float* corm, int dim, float thv, double &pm, int snpn) {
    vector<int> extractId_tmp, extractId;
    vector<float> tv;
    double top_tmp, top_current;
    int extractLen;
    int n,n_tmp;

    // init pm
    pm = 1;

    for (int i = 0; i < snpn; i++) { // extract each line of tm
        extractId.clear();
        extractId_tmp.clear();
        tv.clear();
        for (int j = 0; j < dim; j++) {
            if (abs(tm[i * dim + j]) > thv) {
                extractId.push_back(j);
            }
        }
        extractLen = extractId.size();
        if (extractLen == 0) {
            pm = min(pm, 1.0);
            continue;
        }

        top_current = 1;
        n = 1; // n indicate the number of selected items
        for (int j = 0; j < dim; j++) {
            tv.push_back(tm[i * dim + j]);
        }
        sort_indexes_abs_decrease(tv, extractId);
        for (int j = 0; j < extractLen; j++) {
            extractId_tmp.push_back(extractId[j]);
        }
        trtvf(tv, extractId, n, corm, dim, top_tmp);
        for (int j = 0; j < extractLen; j++) {
            extractId_tmp[j] = extractId[j];
        }

        n_tmp = n - 1;
        while (top_tmp < top_current) {
            if (n >= extractLen) {
                break;
            }
            top_current = top_tmp;
            float t_tmp = extractId[n_tmp];
            for (int j = n_tmp; j > n - 1; j--) {
                extractId[j] = extractId[j - 1];
            }
            extractId[n - 1] = t_tmp;

            n++;

            trtvf(tv, extractId, n, corm, dim, top_tmp);
            for (int j = 0; j < extractLen; j++) {
                extractId_tmp[j] = extractId[j];
            }
            n_tmp = n - 1;
            while (top_tmp >= top_current) {
                n_tmp++;
                if (n_tmp >= extractLen) {
                    break;
                }
                extractId_tmp[n - 1] = extractId_tmp[n_tmp];
                trtvf(tv, extractId_tmp, n, corm, dim, top_tmp);
            }
        }
        pm = min(pm, top_current);
    }
    return(0);
}

void timeElapse(timeval &time_start, string stepName) {
    struct timeval time_end;
    gettimeofday(&time_end,NULL);
    cout << stepName + " step using time:" 
    << (time_end.tv_sec-time_start.tv_sec)+(time_end.tv_usec-time_start.tv_usec)/1000000.0 
    << "s" << endl;
    gettimeofday(&time_start,NULL);
}
