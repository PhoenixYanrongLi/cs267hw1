const char* dgemm_desc = "Naive, reordered, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */   


//Naively change the order of the loop.
void square_dgemm (int n, double* A, double*B, double* C)
{
  for (int j = 0; j < n; j++)
    for (int k = 0; k < n; k++)
      for (int i = 0; i < m; i++)
        c[i+j*n] += A[i+k*n] * B[k+j*n];
}

