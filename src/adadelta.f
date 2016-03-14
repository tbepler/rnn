
module adadelta_module
  use iso_c_binding
  implicit none

  private

  public :: dadadelta
  public :: sadadelta
  
contains

#define FLOAT_T real(c_double)
#define PREFIX d
#include "adadelta.fi"
#undef FLOAT_T
#undef PREFIX

#define FLOAT_T real(c_float)
#define PREFIX s
#include "adadelta.fi"
#undef FLOAT_T
#undef PREFIX
  
end module adadelta_module
