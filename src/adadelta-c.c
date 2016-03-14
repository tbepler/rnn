
#include <math.h>

void adadelta(double rho, double eps, int n, const double* g, double* x, double* eg, double* dx){

  for(int i = 0 ; i < n ; ++i){
    eg[i] = rho*eg[i] + (1-rho)*g[i]*g[i];
    double dxi = sqrt(dx[i] + eps)/sqrt(eg[i] + eps)*g[i];
    dx[i] = rho*dx[i] + (1-rho)*dxi*dxi;
    x[i] -= dxi;
  }
  
}
