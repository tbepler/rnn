
module embed_convolution_module
  use iso_c_binding
  implicit none

  private

  public :: idemconvfw
  public :: ldemconvfw
  public :: isemconvfw
  public :: lsemconvfw
  public :: idemconvbw
  public :: ldemconvbw
  public :: isemconvbw
  public :: lsemconvbw
  
contains

#define FLOAT_T real(c_double)
#define GEMM dgemm
#define INT_T integer(c_int)
#define PREFIX id
#include "embed-convolution.fi"
#undef INT_T
#undef PREFIX
#define INT_T integer(c_long)
#define PREFIX ld
#include "embed-convolution.fi"
#undef INT_T
#undef FLOAT_T
#undef GEMM
#undef PREFIX

#define FLOAT_T real(c_float)
#define GEMM sgemm
#define INT_T integer(c_int)
#define PREFIX is
#include "embed-convolution.fi"
#undef INT_T
#undef PREFIX
#define INT_T integer(c_long)
#define PREFIX ls
#include "embed-convolution.fi"
#undef INT_T
#undef FLOAT_T
#undef GEMMx
#undef PREFIX
  
end module embed_convolution_module
