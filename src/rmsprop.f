
module rmsprop_module
  use iso_c_binding
  implicit none

  private

  public :: drmsprop
  public :: srmsprop
  
contains

#define FLOAT_T real(c_double)
#define PREFIX d
#include "rmsprop.fi"
#undef FLOAT_T
#undef PREFIX

#define FLOAT_T real(c_float)
#define PREFIX s
#include "rmsprop.fi"
#undef FLOAT_T
#undef PREFIX
  
end module rmsprop_module
