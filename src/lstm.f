
module lstm_module
  use iso_c_binding
  implicit none

  private

  public :: dlstmfw
  public :: slstmfw
  public :: dlstmbw
  public :: slstmbw
  public :: dlstmencbw
  public :: slstmencbw
  public :: dlstmrfw
  public :: slstmrfw
  public :: dlstmrbw
  public :: slstmrbw
  !public :: dbilstmfw
  !public :: sbilstmfw
  !public :: dbilstmbw
  !public :: sbilstmbw
  
contains

#define FLOAT_T real(c_double)
#define GEMM dgemm
#define PREFIX d
#include "lstm.fi"
#undef FLOAT_T
#undef GEMM
#undef PREFIX

#define FLOAT_T real(c_float)
#define GEMM sgemm
#define PREFIX s
#include "lstm.fi"
#undef FLOAT_T
#undef GEMM
#undef PREFIX
  
end module lstm_module



