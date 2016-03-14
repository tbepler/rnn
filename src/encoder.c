
#include <string.h>
#include <cblas.h>
//#include <stdio.h>

static inline void mask(int m, double* y){
  for(int i = 0 ; i < m ; ++i){
    y[i] = 0;
  }
}

void encoder_forward( int m, int b, const double* self, int k, const int* x, int ldx, double* y ){
  //m is output size
  //x is vector of indices indicating which word
  for(int i = 0 ; i < k ; ++i){
    for(int j = 0 ; j < b ; ++j){
      //check for mask
      if(x[i*ldx+j] < 0){
	mask(m, y+(i*b+j)*m);
      }else{
	//copy the x[i*ldx+j]th column of self into (i*b+j)th column of y
	memcpy(y+(i*b+j)*m, self+x[i*ldx+j]*m, m*sizeof(*self));
      }
    }
  }
}

void encoder_backward( int m, int b, int k, const int* x, int ldx, const double* dy
		       , double* dself ){

  for(int i = 0 ; i < k ; ++i){
    for(int j = 0 ; j < b ; ++j){
      cblas_daxpy(m, 1, dy+(i*b+j)*m, 1, dself+x[i*ldx+j]*m, 1);
    }
  }
  
}


