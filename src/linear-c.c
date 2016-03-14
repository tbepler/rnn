
#include <string.h>
#include <cblas.h>


void linear_forward( int m, int n, const double* self, int k, const double* x, double* y ){
  //m is output size
  //n is input size
  //first m elements of self are bias
  //copy bias into y
  for( int i = 0 ; i < k ; ++i ){
    memcpy(y+i*m, self, m*sizeof(*self)); 
  }
  //compute y <- self*x + y
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1, self+m, m, x, n, 1, y, m);
}

void linear_backward(int m, int n, const double* self, int k, const double* x, const double* dy
                    , double* dx, double* dself){

  //fill first column of dself with row sums of dy (sum of derivatives of bias)
  for(int i = 0 ; i < k ; ++i){
    cblas_daxpy(m, 1, dy+i*m, 1, dself, 1);
  }
  //add each column of x to every row of dself
  for(int i = 0 ; i < k ; ++i){
    for(int j = 0 ; j < n ; ++j){
      double xij = x[i*n+j];
      cblas_daxpy(m, xij, dy+i*m, 1, dself+m+j*m, 1);
    }
  }

  //compute dx <- selfT*dy
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, k, m, 1, self+m, m, dy, m, 0, dx, n);
  
}



