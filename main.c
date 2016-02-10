#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include"dgemm.h"


int main()
{
  double A[25] = {1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5};
  double B[25] = {1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5};
  double C[25] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  square_dgemm (5, A, B, C);
  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 5; j++)
      printf("%f ", C[i+j*5]);
    printf("\n");
  }

  return 0;

}