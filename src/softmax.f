
module softmax_module
  use iso_c_binding
  implicit none

  private

  public :: dsoftmaxfw
  public :: ssoftmaxfw
  public :: dicentfw
  public :: dlcentfw
  public :: sicentfw
  public :: slcentfw
  public :: dientmaxbw
  public :: dlentmaxbw
  public :: sientmaxbw
  public :: slentmaxbw
  
contains

#define FLOAT_T real(c_double)
#define PREFIX d
#include "softmax.fi"
#undef FLOAT_T
#undef PREFIX

#define FLOAT_T real(c_float)
#define PREFIX s
#include "softmax.fi"
#undef FLOAT_T
#undef PREFIX
  
end module softmax_module
