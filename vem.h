#ifndef VEM_H_INCLUDED
#define VEM_H_INCLUDED

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <global.h>
#include <workers.h>
#include <data.h>
#include <gmm.h>

#define vget gsl_vector_get
#define mget gsl_matrix_get
#define vset gsl_vector_set
#define mset gsl_matrix_set

#define VBGMMMAXITER 120
#define THRES 0.0000000001
#define PI 3.14159265358979323846

typedef struct VBGMM
{
    int K;
    int dim;
    gsl_vector *alpha;
    gsl_vector *beta;
    gsl_vector *v;
    gsl_matrix **W;
    gsl_matrix *m;
    gsl_matrix **S;
    gsl_vector *pi;
    gsl_matrix *xbarra;
    gsl_vector *L;
} VBGMM;

void vbg_save(char *fname,VBGMM *m);
void vbg_delete(VBGMM *m);
double determinante (gsl_matrix *m);

VBGMM* vbg_alloc(number k, number d);
void vbg_save(char *fname,VBGMM *m);
void vbg_delete(VBGMM *m);

double somatorio (gsl_vector *v);
gsl_matrix* inver (gsl_matrix *m);

void vbg_vem (VBGMM *vbg, gmm *gm, data *dado, double alpha0, double beta0, double v0, gsl_vector *m0, gsl_matrix *W0);


gsl_matrix **invS;

#endif // VEM_H_INCLUDED
