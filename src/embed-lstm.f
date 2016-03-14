
module embed_lstm_module
  use iso_c_binding
  implicit none

  private

  public :: diemlstmfw
  public :: dlemlstmfw
  public :: siemlstmfw
  public :: slemlstmfw
  public :: diemlstmbw
  public :: dlemlstmbw
  public :: siemlstmbw
  public :: slemlstmbw
  public :: diemlstmencbw
  public :: dlemlstmencbw
  public :: siemlstmencbw
  public :: slemlstmencbw
  
contains

#define FLOAT_T real(c_double)
#define GEMM dgemm
#define IDX_T integer(c_int)
#define PREFIX di
#include "embed-lstm.fi"
#undef IDX_T
#undef PREFIX
#define IDX_T integer(c_long)
#define PREFIX dl
#include "embed-lstm.fi"
#undef IDX_T
#undef PREFIX
#undef FLOAT_T
#undef GEMM
#undef PREFIX

#define FLOAT_T real(c_float)
#define GEMM sgemm
#define IDX_T integer(c_int)
#define PREFIX si
#include "embed-lstm.fi"
#undef IDX_T
#undef PREFIX
#define IDX_T integer(c_long)
#define PREFIX sl
#include "embed-lstm.fi"
#undef IDX_T
#undef PREFIX
#undef FLOAT_T
#undef GEMM
#undef PREFIX
  
end module embed_lstm_module



