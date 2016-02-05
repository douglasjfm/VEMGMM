/* Variational Expectation Maximization for Gaussian Mixture Models.
Copyright (C) 2012-2016 Douglas Medeiros

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details. */

#include "vem.h"

/*!
    Program for reading the data and train a Variational Bayesian GMM
    argv[1] -> unsigned int K, the number of gaussian components
    argv[2] -> char* name for the data file (see oldfaithfull.txt)

    The program will output a file with a structure VBGMM saved at a file named
    'vgmm_out.txt'.
*/

int main(int argc, char *argv[])
{
    VBGMM *vbg;
    int numK;
    workers *pool=NULL;
    data *dado = NULL;
    char ftrn[150];
    gmm *gm;

    gsl_vector *m0;
    gsl_matrix *W0;
    double alpha0,beta0,v0;

    numK = atoi(argv[1]);
    strcpy(ftrn,argv[2]);
    pool = workers_create(1);
    dado = feas_load(ftrn,pool);

    vbg = vbg_alloc(numK,dado->dimension);

    gm = gmm_initialize(dado,vbg->K);
    gmm_EMtrain(dado,gm,pool);

    vbg_vem(vbg,gm,dado,alpha0,beta0,v0,m0,W0);
    vbg_save("vgmm_out.txt",vbg);
    vbg_delete(vbg);

    return 0;
}
