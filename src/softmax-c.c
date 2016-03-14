
#include <cblas.h>
#include <math.h>
#include <string.h>

void softmax_forward(int m, int n, const double* x, double* y){

  for(int j = 0 ; j < n ; ++j){
    double sum = 0;
    for(int i = 0 ; i < m ; ++i){
      double v = exp(x[i+j*m]);
      y[i+j*m] = v;
      sum += v;
    }
    cblas_dscal(m, 1/sum, y+j*m, 1);
  }
  
}

static inline void mask(int m, double* dx){
  for(int i = 0 ; i < m ; ++i){
    dx[i] = 0;
  }
}

void softmax_cross_entropy_backward(int m, int n, const double* yh
				    , const int* y, double* dx){

  //first copy yh to dx
  memcpy(dx, yh, m*n*sizeof(*dx));

  for(int j = 0 ; j < n ; ++j){
    int i = y[j];
    //check for mask
    if(i < 0){
      mask(m, dx+j*m);
    }else{
      dx[i+j*m] -= 1;
    }
  }
  
}


