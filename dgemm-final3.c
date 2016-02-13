#include<immintrin.h>
#include<stdlib.h>
#include<string.h>


const char* dgemm_desc = "SSE blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 40
#endif
 
#define min(a,b) (((a)<(b))?(a):(b))


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B'
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static int adjustSize (int size, int step)
{
  int newSize;
  if (size % step == 0) return size;
  else{
    newSize = size + step - size % step;
  }

  return newSize;

}

static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
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
      b7 = _mm256_set1_pd(B[k+1+(j+3)*K]);

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

//  if(width % 128 == 0 || width % 128 == 127) widthAdjusted += 8;
//  if(height % 128 == 0 || height % 128 == 127) heightAdjusted += 8;  


  for (int j = 0; j < width; j++)
  {
    memcpy(padded_A+j*heightAdjusted, A+j*lda, height * sizeof(double));
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

  //if(width % 128 == 0 || width % 128 == 127) widthAdjusted += 8;
  //if(height % 128 == 0 || height % 128 == 127) heightAdjusted += 8;  


  for (int j = 0; j < width; j++){
    memcpy(A+j*lda, padded_A+j*heightAdjusted, height * sizeof(double));
  }
}


void square_dgemm (int lda, double* A, double* B, double* C)
{
  
  int blockRowSize = 12;
  int blockColSize = 220;
  int block3rdSize = 220;

  double* block_A = (double*)malloc(((blockColSize * block3rdSize)+100)*sizeof(double));
  double* block_B = (double*)malloc(((block3rdSize * lda)+1000)*sizeof(double));
  double* block_C = (double*)malloc(((blockColSize * blockRowSize)+100)*sizeof(double));

  for(int k = 0; k < lda; k += block3rdSize){
    int K = min(block3rdSize, lda - k);
    int adjusted_K = adjustSize(K, 4);
    copy_block(lda, K, lda, B+k, block_B);
    
    for(int i = 0; i < lda; i += blockColSize){
      int M = min(blockColSize, lda - i);
      int adjusted_M = adjustSize(M, 4);
      copy_block(lda, M, K, A+i+k*lda, block_A);

      for(int j = 0; j < lda; j += blockRowSize){
        int N = min(blockRowSize, lda - j);
        int adjusted_N = adjustSize(N, 4);
        copy_block(lda, M, N, C+i+j*lda, block_C);

        do_block(lda, adjusted_M, adjusted_N, adjusted_K, block_A, block_B + j * adjusted_K, block_C);
        
        store_block(lda, M, N, C+i+j*lda, block_C);

      }
      
    }

  }

  free(block_A);
  free(block_B);
  free(block_C);

}
