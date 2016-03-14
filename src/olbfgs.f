
module olbfgs_module
  use iso_c_binding
  implicit none

  private

  public :: dolbfgs
  public :: solbfgs
  
contains

#define FLOAT_T real(c_double)
#define PREFIX d
#include "olbfgs.fi"
#undef FLOAT_T
#undef PREFIX

#define FLOAT_T real(c_float)
#define PREFIX s
#include "olbfgs.fi"
#undef FLOAT_T
#undef PREFIX
  
end module olbfgs_module
