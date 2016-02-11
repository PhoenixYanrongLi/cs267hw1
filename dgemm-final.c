#include<immintrin.h>
#include<stdlib.h>
#include<string.h>


const char* dgemm_desc = "SSE blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
 
#define min(a,b) (((a)<(b))?(a):(b))

#define SSE_SIZE 4
#define UNROLL_SIZE 4

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
  __m256d a0, a1, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3;

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
        a0 = _mm256_load_pd(A+i+k*M);
        a1 = _mm256_load_pd(A+i+(k+1)*M);

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
        c3 = _mm256_add_pd(c3, _mm256_mul_pd(a1,b7));

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

  //__m256d tmp1, tmp2, tmp3, tmp4;
  int heightAdjusted = adjustSize(height, 4);
  int widthAdjusted = adjustSize(width, 4);

  for (int j = 0; j < width; j++)
  {
    memcpy(A+j*lda, padded_A+j*heightAdjusted, height * sizeof(double));
    for (int i = height; i < heightAdjusted; i++)
      padded_A[i+j*heightAdjusted] = 0.0;
  }
  
  for (int j = width; j < widthAdjusted; j++)
    for (int i = 0; i < height; i++)
      padded_A[i+j*heightAdjusted] = 0.0;
}



/* Store the calculated result from the adjusted-size block to the original matrix */
static void store_block(int lda, int height, int width, double* A, double* padded_A)
{
  int heightAdjusted = adjustSize(height, 4);
  int widthAdjusted = adjustSize(width, 4);

  for (int j = 0; j < width; j++){
    memcpy(padded_A+j*heightAdjusted, A+j*lda, height * sizeof(double));
  }
}






void square_dgemm (int lda, double* A, double* B, double* C)
{
  
  int adjusted_lda = adjustSize(lda, 4);

  double* padded_A = (double*)malloc((adjusted_lda * adjusted_lda+1000)*sizeof(double));
  double* padded_B = (double*)malloc((adjusted_lda * adjusted_lda+1000)*sizeof(double));
  double* padded_C = (double*)malloc((adjusted_lda * adjusted_lda+1000)*sizeof(double));

  copy_block(lda, lda, lda, A, padded_A);
  copy_block(lda, lda, lda, B, padded_B);
  copy_block(lda, lda, lda, C, padded_C);


  /* For each block-row of A */ 
  for (int k = 0; k < adjusted_lda; k += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < adjusted_lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int i = 0; i < adjusted_lda; i += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, adjusted_lda-i);
        int N = min (BLOCK_SIZE, adjusted_lda-j);
        int K = min (BLOCK_SIZE, adjusted_lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, padded_A + i + k*lda, padded_B + k + j*lda, padded_C + i + j*lda);
      }
  
  store_block(lda, lda, lda, C, padded_C);

  free(padded_A);
  free(padded_B);
  free(padded_C);

}
