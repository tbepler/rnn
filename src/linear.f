
module linear_module
  use iso_c_binding
  implicit none

  private

  public :: dlinearfw
  public :: slinearfw
  public :: dlinearbw
  public :: slinearbw
  
contains

#define FLOAT_T real(c_double)
#define GEMM dgemm
#define PREFIX d
#include "linear.fi"
#undef FLOAT_T
#undef GEMM
#undef PREFIX

#define FLOAT_T real(c_float)
#define GEMM sgemm
#define PREFIX s
#include "linear.fi"
#undef FLOAT_T
#undef GEMM
#undef PREFIX
  
end module linear_module
