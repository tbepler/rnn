
module activations
  implicit none
  
contains

  pure real(4) function float_fast_tanh(x)
    real(4), intent(in) :: x
    float_fast_tanh = x/(1+abs(x))
  end function float_fast_tanh
  
  pure real(8) function double_fast_tanh(x)
    real(8), intent(in) :: x
    double_fast_tanh = x/(1+abs(x))
  end function double_fast_tanh

  pure real(4) function float_fast_sigmoid(x)
    real(4), intent(in) :: x
    float_fast_sigmoid = (float_fast_tanh(x) + 1)/2
  end function float_fast_sigmoid
  
  pure real(8) function double_fast_sigmoid(x)
    real(8), intent(in) :: x
    double_fast_sigmoid = (double_fast_tanh(x) + 1)/2
  end function double_fast_sigmoid

  subroutine float_lstm_activations(m, b, W, S, Y)
    integer, intent(in) :: m, b
    real(4), intent(in) :: W(4*m)
    real(4), intent(inout) :: S(6*m,2*b)
    real(4), intent(out) :: Y(m,2*b)
    integer :: i, j

    do concurrent (i=1:m, j=b+1:2*b)
       S(i,j) = float_fast_sigmoid(S(i,j) + W(i)) !Input gate
       S(m+i,j) = float_fast_sigmoid(S(m+i,j) + W(m+i)) !Forget gate
       S(2*m+i,j) = float_fast_sigmoid(S(2*m+i,j) + W(2*m+i)) !Output gate
       S(3*m+i,j) = float_fast_tanh(S(3*m+i,j) + W(3*m+i)) !Activation gate
       S(4*m+i,j) = S(i,j)*S(3*m+i,j) + S(m+i,j)*S(4*m+i,j-b) !Update C
       S(5*m+i,j) = float_fast_tanh(S(4*m+i, j)) !Ct
       Y(i,j) = S(2*m+i,j)*S(5*m+i,j) !Y
    end do
    
  end subroutine float_lstm_activations
  
  subroutine double_lstm_activations(m, b, W, S, Y)
    integer, intent(in) :: m, b
    real(8), intent(in) :: W(4*m)
    real(8), intent(inout) :: S(6*m,2*b)
    real(8), intent(out) :: Y(m,2*b)
    integer :: i, j

    do concurrent (i=1:m, j=b+1:2*b)
       S(i,j) = double_fast_sigmoid(S(i,j) + W(i)) !Input gate
       S(m+i,j) = double_fast_sigmoid(S(m+i,j) + W(m+i)) !Forget gate
       S(2*m+i,j) = double_fast_sigmoid(S(2*m+i,j) + W(2*m+i)) !Output gate
       S(3*m+i,j) = double_fast_tanh(S(3*m+i,j) + W(3*m+i)) !Activation gate
       S(4*m+i,j) = S(i,j)*S(3*m+i,j) + S(m+i,j)*S(4*m+i,j-b) !Update C
       S(5*m+i,j) = double_fast_tanh(S(4*m+i, j)) !Ct
       Y(i,j) = S(2*m+i,j)*S(5*m+i,j) !Y
    end do
    
  end subroutine double_lstm_activations

  !This is slower in MKL but faster in OpenBLAS...
  subroutine float_lstm_small_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
    integer, intent(in) :: m, n, b, p, ldx, ldy, lds
    real(4), intent(in) :: X(ldx,b*p), W(4*m,m+n+1)
    real(4), intent(inout) :: Y(ldy,b*p+b), S(lds,b*p+b)
    integer :: i, offset
    real(4), parameter :: one = 1, zero = 0
   
    !compute sequential activations
    do i=0,p-1
       offset = i*b
       call sgemm('N', 'N', 4*m, b, n, one, W(:,2:), 4*m, X(:,offset+1:), ldx, zero &
                , S(:,offset+b+1:), lds)
       !multiply Y(prev) by the weights corresponding to Y and add to S
       call sgemm('N', 'N', 4*m, b, m, one, W(:,n+2:), 4*m, Y(:,offset+1:), ldy, one &
                , S(:,offset+b+1:), lds)
       !compute activations for each unit in each batch
       call float_lstm_activations(m, b, W, S(:,offset+1:), Y(:,offset+1:))
    end do
    
  end subroutine float_lstm_small_batch
  
  !This is slower in MKL but faster in OpenBLAS...
  subroutine double_lstm_small_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
    integer, intent(in) :: m, n, b, p, ldx, ldy, lds
    real(8), intent(in) :: X(ldx,b*p), W(4*m,m+n+1)
    real(8), intent(inout) :: Y(ldy,b*p+b), S(lds,b*p+b)
    integer :: i, offset
    real(8), parameter :: one = 1, zero = 0
   
    !compute sequential activations
    do i=0,p-1
       offset = i*b
       call dgemm('N', 'N', 4*m, b, n, one, W(:,2:), 4*m, X(:,offset+1:), ldx, zero &
                , S(:,offset+b+1:), lds)
       !multiply Y(prev) by the weights corresponding to Y and add to S
       call dgemm('N', 'N', 4*m, b, m, one, W(:,n+2:), 4*m, Y(:,offset+1:), ldy, one &
                , S(:,offset+b+1:), lds)
       !compute activations for each unit in each batch
       call double_lstm_activations(m, b, W, S(:,offset+1:), Y(:,offset+1:))
    end do
    
  end subroutine double_lstm_small_batch

  subroutine float_lstm_large_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
    integer, intent(in) :: m, n, b, p, ldx, ldy, lds
    real(4), intent(in) :: X(ldx,b*p), W(4*m,m+n+1)
    real(4), intent(inout) :: Y(ldy,b*p+b), S(lds,b*p+b)
    integer :: i, offset
    real(4), parameter :: one = 1, zero = 0

    !multiply X by the weights corresponding to X and write to S
    call sgemm('N', 'N', 4*m, b*p, n, one, W(:,2:), 4*m, X, ldx, zero, S(:,b+1:), lds)
    
    !compute sequential activations
    do i=0,p-1
       offset = i*b
       !multiply Y(prev) by the weights corresponding to Y and add to S
       call sgemm('N', 'N', 4*m, b, m, one, W(:,n+2:), 4*m, Y(:,offset+1:), ldy, one &
                , S(:,offset+b+1:), lds)
       !compute activations for each unit in each batch
       call float_lstm_activations(m, b, W, S(:,offset+1:), Y(:,offset+1:))
    end do
    
  end subroutine float_lstm_large_batch
  
  subroutine double_lstm_large_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
    integer, intent(in) :: m, n, b, p, ldx, ldy, lds
    real(8), intent(in) :: X(ldx,b*p), W(4*m,m+n+1)
    real(8), intent(inout) :: Y(ldy,b*p+b), S(lds,b*p+b)
    integer :: i, offset
    real(8), parameter :: one = 1, zero = 0

    !multiply X by the weights corresponding to X and write to S
    call dgemm('N', 'N', 4*m, b*p, n, one, W(:,2:), 4*m, X, ldx, zero, S(:,b+1:), lds)
    
    !compute sequential activations
    do i=0,p-1
       offset = i*b
       !multiply Y(prev) by the weights corresponding to Y and add to S
       call dgemm('N', 'N', 4*m, b, m, one, W(:,n+2:), 4*m, Y(:,offset+1:), ldy, one &
                , S(:,offset+b+1:), lds)
       !compute activations for each unit in each batch
       call double_lstm_activations(m, b, W, S(:,offset+1:), Y(:,offset+1:))
    end do
    
  end subroutine double_lstm_large_batch
   
end module activations

subroutine slstm(m, n, b, p, X, ldx, W, Y, ldy, S, lds) BIND(C)
  use activations
  implicit none

  ! m is output size
  ! n is input size
  ! b is number of batches
  ! p is length of sequence
  integer, intent(in), value :: m, n, b, p, ldx, ldy, lds
  real(4), intent(in) :: X(ldx,b*p), W(4*m,m+n+1)
  real(4), intent(inout) :: Y(ldy,b*p+b), S(lds,b*p+b)

  !if (b < 16) then
  !   call lstm_small_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
  !else
  !   call lstm_large_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
  !end if
  call float_lstm_large_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
  
end subroutine slstm

subroutine dlstm(m, n, b, p, X, ldx, W, Y, ldy, S, lds) BIND(C)
  use activations
  implicit none

  ! m is output size
  ! n is input size
  ! b is number of batches
  ! p is length of sequence
  integer, intent(in), value :: m, n, b, p, ldx, ldy, lds
  real(8), intent(in) :: X(ldx,b*p), W(4*m,m+n+1)
  real(8), intent(inout) :: Y(ldy,b*p+b), S(lds,b*p+b)

  !if (b < 16) then
  !   call lstm_small_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
  !else
  !   call lstm_large_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
  !end if
  call double_lstm_large_batch(m, n, b, p, X, ldx, W, Y, ldy, S, lds)
  
end subroutine dlstm
  

