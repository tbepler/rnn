
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <omp.h>

//this fixes weird complex.h inclusion issue
#undef I

static inline double sigmoid(double x){
  return 1.0/(1.0 + exp(-x));
}

static inline double fast_tanh(double x){
  return x/(1 + fabs(x));
}

static inline double fast_tanh_derivative(double y){
  double x = y/(1 - fabs(y));
  double fx = fabs(x) + 1;
  return 1/(fx*fx);
}

static inline double fast_sigmoid(double x){
  return (fast_tanh(x) + 1)/2;
}

static inline double fast_sigmoid_derivative(double y){
  return fast_tanh_derivative(2*y - 1)/2;
}

//typedef double v4d __attribute__((vector_size(sizeof(double)*4)));
//typedef double v4d __attribute__((ext_vector_type(4)));

static void lstm_activations(int m, double* restrict I, double* restrict F, double* restrict O
			     , double* restrict G, double* restrict C, double* restrict Ct
			     , double* restrict H, const double* restrict Cprev ){

  //multiple loops allows for vectorization
  for(int i = 0 ; i < m ; ++i){
    I[i] = fast_sigmoid(I[i]);
    F[i] = fast_sigmoid(F[i]);
    O[i] = fast_sigmoid(O[i]);
  }
  
  for(int i = 0 ; i < m ; ++i){
    G[i] = fast_tanh(G[i]);
    C[i] = I[i]*G[i] + F[i]*Cprev[i];
  }
  
  for(int i = 0 ; i < m ; ++i){
    Ct[i] = fast_tanh(C[i]);
    H[i] = O[i]*Ct[i];
  }

  //#pragma clang loop vectorize(enable)
  /*for(int i = 0 ; i < m ; ++i){
    //I[i] = sigmoid(I[i]);
    //F[i] = sigmoid(F[i]);
    //O[i] = sigmoid(O[i]);
    //G[i] = tanh(G[i]);
    I[i] = fast_sigmoid(I[i]);
    F[i] = fast_sigmoid(F[i]);
    O[i] = fast_sigmoid(O[i]);
    G[i] = fast_tanh(G[i]);
    C[i] = I[i]*G[i] + F[i]*Cprev[i];
    //Ct[i] = tanh(C[i]);
    Ct[i] = fast_tanh(C[i]);
    H[i] = O[i]*Ct[i];
    }*/
}

static void lstm_batch_activations(int m, int n, int b
				   , const double* restrict X, int ldx
				   , const double* restrict W
				   , const double* restrict Wx
				   , const double* restrict Wy
				   , const double* restrict Yprev
				   , double* restrict Y, int ldy
				   , const double* restrict Cprev
				   , double* restrict S, int lds ){

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4*m, b, n, 1, Wx, 4*m
  	      , X, ldx, 0, S, lds);
  //first compute IFOG += W*Yprev
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4*m, b, m, 1, Wy, 4*m
  	      , Yprev, ldy, 1, S, lds);

  //compute nonlinearity of gates and the ouput
  //#pragma omp parallel for
  for(int j = 0 ; j < b ; ++j){
    //cblas_dgemv(CblasColMajor, CblasNoTrans, 4*m, n, 1, Wx, 4*m, X+j*ldx, 1, 0, S+j*lds, 1);
    //cblas_dgemv(CblasColMajor, CblasNoTrans, 4*m, m, 1, Wy, 4*m, Yprev+j*ldy, 1, 1, S+j*lds, 1);
    cblas_daxpy(4*m, 1, W, 1, S+j*lds, 1);
    double* restrict I = S + j*lds;
    double* restrict F = I + m;
    double* restrict O = F + m;
    double* restrict G = O + m;
    double* restrict C = G + m;
    double* restrict Ct = C + m;
    double* restrict H = Y + j*ldy;
    const double* restrict Cprev_batch = Cprev + j*lds;
    lstm_activations(m, I, F, O, G, C, Ct, H, Cprev_batch);
    /*for(int i = 0 ; i < m ; ++i){
      //I[i] = sigmoid(I[i]);
      //F[i] = sigmoid(F[i]);
      //O[i] = sigmoid(O[i]);
      //G[i] = tanh(G[i]);
      I[i] = fast_sigmoid(I[i]);
      F[i] = fast_sigmoid(F[i]);
      O[i] = fast_sigmoid(O[i]);
      G[i] = fast_tanh(G[i]);
      C[i] = I[i]*G[i] + F[i]*Cprev_batch[i];
      //Ct[i] = tanh(C[i]);
      Ct[i] = fast_tanh(C[i]);
      H[i] = O[i]*Ct[i];
      }*/
  }
  
}

void lstm_batch_forward(int m, int n, int b, int p, const double* X, int ldx, const double* W
			, const double* Yprev, double* Y, int ldy, const double* Sprev, double* S
			, int lds){

  //first write the bias into each IFOG column - there are b columns for each p
  //for(int i = 0 ; i < b*p ; ++i){
  //  memcpy(S + i*lds, W, 4*m*sizeof(*W)); //bias is first 4m entries of W
  //}
  //next compute IFOG += Wx*X
  const double* Wx = W + 4*m;
  //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4*m, b*p, n, 1, Wx, 4*m
  //	      , X, ldx, 0, S, lds);

  //compute activations for each time point
  const double* Wy = Wx + 4*m*n;
  const double* Cprev = Sprev + 4*m;
  for(int i = 0 ; i < p ; ++i){
    //for(int j = 0 ; j < b ; ++j){
    //  memcpy(S + j*lds, W, 4*m*sizeof(*W));
    //}
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4*m, b, n, 1, Wx, 4*m
    //		, X, ldx, 1, S, lds);
    lstm_batch_activations(m, n, b, X, ldx, W, Wx, Wy, Yprev, Y, ldy, Cprev, S, lds);
    Yprev = Y;
    Y += b*ldy;
    Cprev = S + 4*m;
    S += b*lds;
    X += b*ldx;
  }
  
}

static void lstm_activation_gradient( int m
				     , const double* restrict dY
				     , const double* restrict Cprev
				     , const double* restrict S
				     , double* restrict dS ){
  const double* restrict I = S;
  const double* restrict F = I + m;
  const double* restrict O = F + m;
  const double* restrict G = O + m;
  const double* restrict Ct = G + 2*m;

  double* restrict dI = dS;
  double* restrict dF = dI + m;
  double* restrict dO = dF + m;
  double* restrict dG = dO + m;
  double* restrict dC = dG + m;
  double* restrict dH = dC + m;
  
  //start by adding the tth column of dY to dH
  cblas_daxpy(m, 1, dY, 1, dH, 1);

  //split these into multiple loops to allow vectorization
  for(int i = 0 ; i < m ; ++i){
    dC[i] += fast_tanh_derivative(Ct[i])*O[i]*dH[i];
    dI[i] = G[i]*dC[i]*fast_sigmoid_derivative(I[i]);
  }
  for(int i = 0 ; i < m ; ++i){
    dF[i] = Cprev[i]*dC[i]*fast_sigmoid_derivative(F[i]);
    dO[i] = Ct[i]*dH[i]*fast_sigmoid_derivative(O[i]);
  }
  for(int i = 0 ; i < m ; ++i){
    dG[i] = I[i]*dC[i]*fast_tanh_derivative(G[i]);
    dC[i] *= F[i];
  }
  
  //backprop output and gates
  /*for(int i = 0 ; i < m ; ++i){
    //dC[i] += (1 - Ct[i]*Ct[i])*O[i]*dH[i];
    dC[i] += fast_tanh_derivative(Ct[i])*O[i]*dH[i];
    //dI[i] = G[i]*dC[i]*I[i]*(1 - I[i]);
    dI[i] = G[i]*dC[i]*fast_sigmoid_derivative(I[i]);
    //dF[i] = Cprev[i]*dC[i]*F[i]*(1 - F[i]);
    dF[i] = Cprev[i]*dC[i]*fast_sigmoid_derivative(F[i]);
    //dO[i] = Ct[i]*dH[i]*O[i]*(1 - O[i]);
    dO[i] = Ct[i]*dH[i]*fast_sigmoid_derivative(O[i]);
    //dG[i] = I[i]*dC[i]*(1 - G[i]*G[i]);
    dG[i] = I[i]*dC[i]*fast_tanh_derivative(G[i]);
    dC[i] *= F[i]; //update dC with the forget gate -> computes dC[t-1]
    }*/
}

static void lstm_batch_gradient(int m, int n, int b
				, const double* X, int ldx
				, double* dX, int lddx
				, const double* W, double* dW
				, const double* Yprev, int ldy
				, const double* dY, int lddy
				, const double* Cprev
				, const double* S, int lds
				, double* dS, int ldds){

  for(int j = 0 ; j < b ; ++j){
    lstm_activation_gradient(m, dY+j*lddy, Cprev+j*lds, S+j*lds, dS+j*ldds);
    //add gates gradient to bias gradient - MOVE TO SEPARATE LOOP TO PARALLELIZE THIS
    cblas_daxpy(4*m, 1, dS+j*ldds, 1, dW, 1);
  }
  //update the gradient of Wx
  double* dWx = dW + 4*m;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 4*m, n, b, 1, dS, ldds
	      , X, ldx, 1, dWx, 4*m);
  //update the gradient of Wy
  double* dWy = dWx + 4*m*n;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 4*m, m, b, 1, dS, ldds
	      , Yprev, ldy, 1, dWy, 4*m);
  //dX = Wx.T * dS
  const double* Wx = W + 4*m;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, b, 4*m, 1, Wx, 4*m
	      , dS, ldds, 0, dX, lddx);
  //dH = Wy.T * dS
  const double* Wy = Wx + 4*m*n;
  double* dH = dS + 5*m;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, b, 4*m, 1, Wy, 4*m
	      , dS, ldds, 0, dH, ldds);
  
}

void lstm_batch_backward(int m, int n, int b, int p, const double* X, int ldx, double* dX, int lddx
			 , const double* W, double* dW, const double* Yprev, const double* Y
			 , int ldy, const double* dY, int lddy, const double* Sprev
			 , const double* S, int lds, double* dS, int ldds){

  //go backwards through time
  const double* Xt = X + (p-1)*b*ldx;
  double* dXt = dX + (p-1)*b*lddx;
  const double* Ytprev = Y + (p-2)*b*ldy;
  const double* dYt = dY + (p-1)*b*lddy;
  const double* Cprev = S + (p-2)*b*lds + 4*m;
  const double* St = S + (p-1)*b*lds;
  for(int i=p-1 ; i > 0 ; --i){
    lstm_batch_gradient(m, n, b, Xt, ldx, dXt, lddx, W, dW, Ytprev, ldy, dYt, lddy, Cprev
			, St, lds, dS, ldds);
    Xt -= b*ldx;
    dXt -= b*lddx;
    Ytprev -= b*ldy;
    dYt -= b*lddy;
    Cprev -= b*lds;
    St -= b*lds;
  }
  //backward for t = 0
  lstm_batch_gradient(m, n, b, X, ldx, dX, lddx, W, dW, Yprev, ldy, dY, lddy, Sprev+4*m
		      , S, lds, dS, ldds);
  
  
}

