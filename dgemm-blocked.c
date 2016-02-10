const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int i = 0; i < M / 4 * 4; i += 4){
    for (int j = 0; j < N; j += 1) {
      __m256d cij = _mm256_loadu_pd(C+i+j*n);

      for(int k = 0; k < K / 4 * 4; k += 4){
        __m256d a0 = _mm256_loadu_pd(A+i+k*n);
        __m256d b0 = _mm256_set1_pd(B[k+j*n]);
        

        __m256d a1 = _mm256_loadu_pd(A+i+(k+1)*n);
        __m256d b1 = _mm256_set1_pd(B[(k+1)+j*n]);
        

        __m256d a2 = _mm256_loadu_pd(A+i+(k+2)*n);
        __m256d b2 = _mm256_set1_pd(B[(k+2)+j*n]);
        

        __m256d a3 = _mm256_loadu_pd(A+i+(k+3)*n);
        __m256d b3 = _mm256_set1_pd(B[(k+3)+j*n]);
        
        cij = _mm256_add_pd(cij, _mm256_mul_pd(a0, b0));
        cij = _mm256_add_pd(cij, _mm256_mul_pd(a1, b1));
        cij = _mm256_add_pd(cij, _mm256_mul_pd(a2, b2));
        cij = _mm256_add_pd(cij, _mm256_mul_pd(a3, b3));

      }

      _mm256_storeu_pd(C+i+j*n, cij);
      if(n % 4 != 0){
        for (int k = n / 4 * 4; k < n; k++){
          C[i+j*n] += A[i+k*n] * B[k+j*n];
        }
      }
    }
  }
  if(n % 4 != 0){
    for(int i = n / 4 * 4; i < n; i++){
      for (int j = 0; j < n; j += 1) {
        
        double tmpcij = C[i+j*n];

        for(int k = 0; k < n / 4 * 4; k += 4){
          tmpcij += A[i+k*n] * B[k+j*n];
          tmpcij += A[i+(k+1)*n] * B[(k+1)+j*n];
          tmpcij += A[i+(k+2)*n] * B[(k+2)+j*n];
          tmpcij += A[i+(k+3)*n] * B[(k+3)+j*n];
        }

        for (int k = n / 4 * 4; k < n; k++){
          tmpcij += A[i+k*n] * B[k+j*n];
        }

        C[i+j*n] = tmpcij;
      }
    }
  }

}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
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
	      do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
