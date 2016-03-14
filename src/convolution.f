
module convolution_module
  use iso_c_binding
  implicit none

  private

  public :: dconvfw
  public :: sconvfw
  public :: dconvbw
  public :: sconvbw
  
contains

#define FLOAT_T real(c_double)
#define GEMM dgemm
#define PREFIX d
#include "convolution.fi"
#undef FLOAT_T
#undef GEMM
#undef PREFIX

#define FLOAT_T real(c_float)
#define GEMM sgemm
#define PREFIX s
#include "convolution.fi"
#undef FLOAT_T
#undef GEMM
#undef PREFIX
  
end module convolution_module
