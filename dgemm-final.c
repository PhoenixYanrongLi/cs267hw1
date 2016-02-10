#include<immintrin.h>
#include<stdlib.h>


const char* dgemm_desc = "SSE blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
 
#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static int adjustSize (int size, int step);
{
  int newSize;
  if (size % step == 0) return size;
  else{
    newSize = size + step - size % step;
  }

  return newSize;

}

static void do_block (int M, int N, int K, double* A, double* B, double* C)
{
  __m256d a0, a1, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1;

  for (int k = 0; k < K; k += 2) {
    for (int j = 0; j < N; j += 4) {
      
      b0 = _mm256_set1_pd(B[k+j*K]);
      b1 = _mm256_set1_pd(B[(k+1)+j*K]);
      b2 = _mm256_set1_pd(B[k+(j+1)*K]);
      b3 = _mm256_set1_pd(B[(k+1)+(j+1)*K]);
      b4 = _mm256_set1_pd(B[k+(j+2)*K]);
      b5 = _mm256_set1_pd(B[(k+1)+(j+2)*K]);
      b6 = _mm256_set1_pd(B[k+(j+3)*K]);
      b7 = _mm256_set1_pd(B[k+(j+3)*K]);

      for (int i = 0; i < M; i += 4) {
        a0 = _mm256_load_pd(A+i+k*K);
        a1 = _mm256_load_pd(A+i+(k+1)*K);

        c0 = _mm256_load_pd(C+i+j*M);
        c1 = _mm256_load_pd(C+i+(j+1)*M);
        c2 = _mm256_load_pd(C+i+(j+2)*M);
        c3 = _mm256_load_pd(C+i+(j+3)*M);

        c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0,b0));
        c0 = _mm256_add_pd(c0, _mm256_mul_pd(a1,b1));
        c1 = _mm256_add_pd(c1, _mm256_mul_pd(a0,b2));
        c1 = _mm256_add_pd(c1, _mm256_mul_pd(a1,b3));
        c2 = _mm256_add_pd(c2, _mm256_mul_pd(a0,b4));
        c2 = _mm256_add_pd(c2, _mm256_mul_pd(a1,b5));
        c3 = _mm256_add_pd(c3, _mm256_mul_pd(a0,b6));
        c3 = _mm256_add_pd(c4, _mm256_mul_pd(a1,b7));

        _mm256_store_pd(C+i+j*M, c0);
        _mm256_store_pd(C+i+(j+1)*M, c1);
        _mm256_store_pd(C+i+(j+2)*M, c2);
        _mm256_store_pd(C+i+(j+3)*M, c3);

      }
    }
  }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  


/* Copy the block and do block padding */
static void copy_block(int lda, int height, int width, double* A, double* padded_A) {

  __m256d tmp1, tmp2, tmp3, tmp4;
  int heightAdjusted = adjustSize(height);
  int widthAdjusted = adjustSize(width);
  
  for (int j = 0; j < width / 4 * 4; j += 4){
    for (int i = 0; i < height / 4 * 4; i += 4){
      tmp1 = _mm256_loadu_pd(A+i+j*lda);
      tmp2 = _mm256_loadu_pd(A+i+(j+1)*lda);
      tmp3 = _mm256_loadu_pd(A+i+(j+2)*lda);
      tmp4 = _mm256_loadu_pd(A+i+(j+3)*lda);

      _mm256_store_pd(padded_A+i+j*heightAdjusted, tmp1);
      _mm256_store_pd(padded_A+i+(j+1)*heightAdjusted, tmp2);
      _mm256_store_pd(padded_A+i+(j+2)*heightAdjusted, tmp3);
      _mm256_store_pd(padded_A+i+(j+3)*heightAdjusted, tmp4);
      
      for (int ii = height / 4 * 4; ii < height; ii++)
        padded_A[ii+j*heightAdjusted] = A[ii+j*lda];
      for (int ii = height; ii < heightAdjusted; ii++)
        padded_A[ii+j*heightAdjusted] = 0.0;

    }

    for (int jj = width / 4 * 4; jj < width; jj++){
      for (int ii = 0; ii < height; ii++)
        padded_A[ii+jj*heightAdjusted] = A[ii+jj*lda];
      for (int ii = height; ii < heightAdjusted; ii++)
        padded_A[ii+jj*heightAdjusted] = 0.0;       
    }

    for(int jj = width; jj < widthAdjusted; jj++){
      for (int ii = 0; ii < heightAdjusted; ii++)
        padded_A[ii+jj*heightAdjusted] = 0.0;
    }
  }
}


/* Store the calculated result from the adjusted-size block to the original matrix */
static void store_block(int lda, int height, int width, double* A, double* padded_A)
{
  __m256d tmp1, tmp2, tmp3, tmp4;
  int heightAdjusted = adjustSize(height);
  int widthAdjusted = adjustSize(width);

  for (int j = 0; j < width / 4 * 4; j += 4){
    for(int i = 0; i < height / 4 * 4; i += 4){
      tmp1 = _mm256_loadu_pd(padded_A+i+j*heightAdjusted);
      tmp2 = _mm256_loadu_pd(padded_A+i+(j+1)*heightAdjusted);
      tmp3 = _mm256_loadu_pd(padded_A+i+(j+2)*heightAdjusted);
      tmp4 = _mm256_loadu_pd(padded_A+i+(j+3)*heightAdjusted);

      _mm256_store_pd(A+i+j*lda, tmp1);
      _mm256_store_pd(A+i+(j+1)*lda, tmp2);
      _mm256_store_pd(A+i+(j+2)*lda, tmp3);
      _mm256_store_pd(A+i+(j+3)*lda, tmp4);
    }
    for(int ii = height / 4 * 4; ii < height; ii++)
      A[ii+j*lda] = padded_A[ii+j*heightAdjusted];
  }
  for (int jj = width / 4 * 4; jj < width; jj++)
    for (int ii = 0; ii < height; ii++)
      A[ii+jj*lda] = padded_A[ii+jj*heightAdjusted];

}


void square_dgemm (int lda, double* A, double* B, double* C)
{
  
  double* padded_A[100000];
  double* padded_B[100000];
  double* padded_C[100000];

  copy_block(lda, lda, lda, A, padded_A);
  copy_block(lda, lda, lda, B, padded_B);
  copy_block(lda, lda, lda, C, padded_C);


  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, padded_A + i + k*lda, padded_B + k + j*lda, padded_C + i + j*lda);
      }
  
  store_block(lda, lda, lda, C, padded_C);

}
