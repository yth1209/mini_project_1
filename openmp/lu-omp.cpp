#include <iostream>
#include <omp.h>
#include <time.h>
#include <chrono>
#include <functional>
#include <map>

using namespace std;

double** multiply_two_array(double* const* A, double* const* B, int matrix_size);
double** mutliply_P_A(double* const* A, const int *P, int matrix_size);
double L1_norm(double* const* A, double* const* B, int matrix_size);

double** init_A(int matrix_size);
int* init_P(int matrix_size);
double** copy_matrix(double* const* A, int matrix_size);

void lu_decomposition(double **A, int *P, int matrix_size);
void decomposed_A_to_L_U(double* const* A, double **L, double **U, int matrix_size);


void usage(const char *name)
{
	std::cout << "usage: " << name
                  << " matrix-size nworkers"
                  << std::endl;
 	exit(-1);
}


int main(int argc, char **argv)
{

  const char *name = argv[0];

  if (argc < 3) usage(name);

  int matrix_size = atoi(argv[1]);

  int nworkers = atoi(argv[2]);

  std::cout << name << ": " 
            << matrix_size << " " << nworkers
            << std::endl;

  omp_set_num_threads(nworkers);

  
  double **A;
  int *P;
  double **L = new double*[matrix_size];
  double **U = new double*[matrix_size];
  double **A_copy = new double*[matrix_size];

  A = init_A(matrix_size);
  P = init_P(matrix_size); 
  A_copy = copy_matrix(A, matrix_size);

  lu_decomposition(A, P, matrix_size);
  decomposed_A_to_L_U(A, L, U, matrix_size);

  double ** LU = multiply_two_array(L, U, matrix_size);
  double ** PA = mutliply_P_A(A_copy, P, matrix_size);
  cout << "L1 Norm Result: " << L1_norm(LU, PA, matrix_size) << endl;

  // Free allocated memory
  for (int i = 0; i < matrix_size; i++) {
    delete[] A[i];
    delete[] L[i];
    delete[] U[i];
  }
  delete[] A;
  delete[] L;
  delete[] U;
  delete[] P;

return 0;
}

//output: AxB
double** multiply_two_array(double* const* A, double* const* B, int matrix_size){
  double **C = new double*[matrix_size];
  for (int i = 0; i < matrix_size; i++){
    C[i] = new double[matrix_size];
  }

  // Transpose B
  double ** B_T = new double*[matrix_size];
  #pragma omp parallel for default(none) shared(B, B_T) firstprivate(matrix_size)
  for (int i = 0; i < matrix_size; i++){
    B_T[i] = new double[matrix_size];
    for (int j = 0; j < matrix_size; j++){
      B_T[i][j] = B[j][i];
    }
  }

  #pragma omp parallel for default(none) shared(A, B_T, C) firstprivate(matrix_size)
  for (int i = 0; i < matrix_size; i++){
    for (int j = 0; j < matrix_size; j++){
      double sum = 0;
      #pragma omp simd
      for (int k = 0; k < matrix_size; k++){
        sum += A[i][k] * B_T[j][k];
      }
      C[i][j] = sum;
    }
  }

  for (int i = 0; i < matrix_size; i++){
    delete[] B_T[i];
  }
  delete[] B_T;

  return C;
}

double** mutliply_P_A(double* const*A, const int *P, int matrix_size){
  double **B = new double*[matrix_size];
  for (int i = 0; i < matrix_size; i++){
    B[i] = new double[matrix_size];
  }

  #pragma omp parallel for default(none) shared(A, B, P) firstprivate(matrix_size)
  for (int i = 0; i < matrix_size; i++){
    for (int j = 0; j < matrix_size; j++){
      B[i][j] = A[P[i]][j];
    }
  }

  return B;
}

double L1_norm(double* const* A, double* const* B, int matrix_size){
  double diff = 0.0;

  #pragma omp parallel for reduction(+:diff) default(none) shared(A, B) firstprivate(matrix_size)
  for (int i = 0; i < matrix_size; i++){
    #pragma omp simd
    for (int j = 0; j < matrix_size; j++){
      diff += abs(A[i][j] - B[i][j]);
    }
  }

  return diff;
}

double** init_A(int matrix_size){
  double **A = new double*[matrix_size];
  
  #pragma omp parallel shared(A)
  {
    struct drand48_data randBuffer;

    srand48_r(chrono::high_resolution_clock::now().time_since_epoch().count(), &randBuffer);
    
    #pragma omp for
    for (int i = 0; i < matrix_size; i++){
      A[i] = new double[matrix_size];
      for (int j = 0; j < matrix_size; j++){
        drand48_r(&randBuffer, &A[i][j]);
      }
    }
  }

  return A;
}

int* init_P(int matrix_size){
  int *P = new int[matrix_size];

  #pragma omp parallel default(none) shared(P) firstprivate(matrix_size)
  {
    #pragma omp for
    for (int i = 0; i < matrix_size; i++){
      P[i] = i;
    }
  }

  return P;
}

double** copy_matrix(double* const* A, int matrix_size){
  double **A_copy = new double*[matrix_size];

  for (int i = 0; i < matrix_size; i++) {
    A_copy[i] = new double[matrix_size];
    for (int j = 0; j < matrix_size; j++) {
      A_copy[i][j] = A[i][j];
    }
  }
  return A_copy;
}

void lu_decomposition(double **A, int *P, int matrix_size) {

  for (int k = 0; k < matrix_size; k++) {
    // Pivoting
    double max = 0.0;
    int k_prime = k;
    
    // Find the maximum element in the k-th column
    #pragma omp parallel default(none) shared(max, k_prime) firstprivate(A, k, matrix_size)
    { 
      // Initialize local variables for each thread
      double local_max = 0.0;
      int local_k_prime = k;

      #pragma omp for nowait
      for (int i = k; i < matrix_size; i++) {
        double abs_A_ik = abs(A[i][k]);
        if (abs_A_ik > local_max) {
          local_max = abs_A_ik;
          local_k_prime = i;
        }
      }

      // Update the global maximum and index if necessary
      #pragma omp critical
      {
        if (local_max > max) {
          max = local_max;
          k_prime = local_k_prime;
        }
      }
    }

    if (max == 0.0) {
      cerr << "Error: Singular matrix" << endl;
      exit(-1);
    }

    
    // Swap rows in P, A, and L
    if(k!=k_prime) {
      swap(P[k], P[k_prime]);
      swap(A[k], A[k_prime]);
    }

    double pivot = A[k][k];
    const double* A_k = A[k];


    #pragma omp parallel for default(none) shared(pivot, A, A_k, k) firstprivate(matrix_size) schedule(dynamic)
    for (int i = k + 1; i < matrix_size; i++) {
      A[i][k] /= pivot;
      double L_ik = A[i][k];

      auto& A_i = A[i];

      #pragma omp simd
      for (int j = k+1; j < matrix_size; j++) {
        A_i[j] -= L_ik * A_k[j];
      }
    }
  }
}

void decomposed_A_to_L_U(double* const* A, double **L, double **U, int matrix_size) {
  // #pragma omp parallel for default(none) shared(A, L, U) firstprivate(matrix_size)
  for (int i = 0; i < matrix_size; i++) {
    L[i] = new double[matrix_size]; 
    U[i] = new double[matrix_size];

    for (int j = 0; j < matrix_size; j++) {
      if (i > j) {
        L[i][j] = A[i][j];
        U[i][j] = 0;
      } else if (i == j) {
        L[i][j] = 1;
        U[i][j] = A[i][j];
      } else {
        L[i][j] = 0;
        U[i][j] = A[i][j];
      }
    }
  }
}