#include<stdio.h>
#include<stdlib.h>
#include<immintrin.h>
#include <math.h>

//Q1-Multiplication de matrices:version naive //

void matmul_naive(float *A, float *B, float *C, int M, int K, int N){

    for(int i = 0; i < M; i++){

        for(int j = 0; j < N; j++){

            C[i*N + j] = 0.0f;

            for(int p = 0; p < K; p++){

                C[i*N + j] += A[i*K +p]*B[p*N + j];
            }
        }
    }
}

//Q2-Multiplication de matrices:version AVX-512//

void matmul_avx512(float *A, float *B, float *C, int M, int K, int N){

    for(int i = 0; i < M; i++){

        for(int j = 0; j < N; j += 16){

            __m512 c_vec = _mm512_setzero_ps();

            for(int p = 0; p < K; p++){

                __m512 b_vec = _mm512_loadu_ps(&B[p*N + j]);

                __m512 a_val = _mm512_set1_ps(A[i*K + p]);

                c_vec = _mm512_fmadd_ps(a_val, b_vec, c_vec);
            }
            _mm512_storeu_ps(&C[i*N + j], c_vec);
        }
    }

}
//Programme principal//
int main(void)
{
    int M = 4;
    int K = 4;
    int N = 32;   // N doit etre multiple de 16 pour AVX-512

    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C_naive = malloc(M * N * sizeof(float));
    float *C_avx = malloc(M * N * sizeof(float));

    if (!A || !B || !C_naive || !C_avx) {
        printf("error allocation\n");
        return 1;
    }

    // Initialisation des matrices
    for (int i = 0; i < M * K; i++)
        A[i] = (float)(i + 1);

    for (int i = 0; i < K * N; i++)
        B[i] = (float)(i % 7 + 1);

    // Calcul naive et vectorise
    matmul_naive(A, B, C_naive, M, K, N);
    matmul_avx512(A, B, C_avx, M, K, N);

  // Affichage
    printf("Resultat C (naïve):\n");
    for (int j = 0; j < N; j++)
        printf("%f ", C_naive[j]);   
    printf("\n");

    printf("Resultat C (AVX-512):\n");
    for (int j = 0; j < N; j++)
        printf("%f ", C_avx[j]);   
    printf("\n");

    // Libération mémoire
    free(A);
    free(B);
    free(C_naive);
    free(C_avx);

    return 0;
}

