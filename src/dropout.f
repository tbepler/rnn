
module dropout_module
  use iso_c_binding
  implicit none

  private

  public :: ddropout
  public :: sdropout
  
contains

#define FLOAT_T real(c_double)
#define PREFIX d
#include "dropout.fi"
#undef FLOAT_T
#undef PREFIX

#define FLOAT_T real(c_float)
#define PREFIX s
#include "dropout.fi"
#undef FLOAT_T
#undef PREFIX
  
end module dropout_module
