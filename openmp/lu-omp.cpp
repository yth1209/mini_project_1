#include <iostream>
#include <omp.h>
#include <time.h>
#include <chrono>
#include <functional>

using namespace std;

void usage(const char *name);
void print_array(double **A, int matrix_size);
void measure_time(function<void()> function);

double** multiply_two_array(double** A, double** B, int matrix_size);
double** permute_A(double **A, int *P, int matrix_size);
double compare_two_array(double** A, double** B, int matrix_size);

double** init_A(int matrix_size);
int* init_P(int matrix_size);
double** init_diagonal_array(int matrix_size);

void lu_decomposition(double **A, int *P, int matrix_size);
void decompose_A_to_L_U(double **A, double **L, double **U, int matrix_size);

int main(int argc, char **argv)
{
  const char *name = argv[0];

  if (argc < 3) usage(name);

  int matrix_size = atoi(argv[1]);
  int nworkers = atoi(argv[2]);

  std::cout << name << ": " << matrix_size << " " << nworkers << std::endl;

  omp_set_num_threads(nworkers);

  double **A = init_A(matrix_size);
  int *P = init_P(matrix_size);
  double **L = init_diagonal_array(matrix_size);
  double **U = init_diagonal_array(matrix_size);

  double **A_copy = new double*[matrix_size];
  for (int i = 0; i < matrix_size; i++) {
    A_copy[i] = new double[matrix_size];
  }
  #pragma omp parallel for
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      A_copy[i][j] = A[i][j];
    }
  }

  measure_time([&]() {
    lu_decomposition(A, P, matrix_size);
  });

  decompose_A_to_L_U(A, L, U, matrix_size);

  // cout << compare_two_array(multiply_two_array(L, U, matrix_size), permute_A(A_copy,P,matrix_size), matrix_size) << endl;


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

void usage(const char *name)
{
	std::cout << "usage: " << name
                  << " matrix-size nworkers"
                  << std::endl;
 	exit(-1);
}

void print_array(double **A, int matrix_size){
  cout << "Print Array Start" << endl;
  for(int i = 0; i < matrix_size; i++){
    for(int j = 0; j < matrix_size; j++){
      cout << A[i][j] << " ";
    }
    cout << endl; 
  }

  cout << "Print Array End" << endl;
}

void measure_time(function<void()> function){
  auto start = std::chrono::high_resolution_clock::now();

  function();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

double** multiply_two_array(double** A, double** B, int matrix_size){
  double **C = new double*[matrix_size];
  for (int i = 0; i < matrix_size; i++){
    C[i] = new double[matrix_size];
  }

  #pragma omp parallel for
  for (int i = 0; i < matrix_size; i++){
    for (int j = 0; j < matrix_size; j++){
      C[i][j] = 0;
      for (int k = 0; k < matrix_size; k++){
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

double** permute_A(double **A, int *P, int matrix_size){
  double **B = new double*[matrix_size];
  for (int i = 0; i < matrix_size; i++){
    B[i] = new double[matrix_size];
  }

  #pragma omp parallel for
  for (int i = 0; i < matrix_size; i++){
    for (int j = 0; j < matrix_size; j++){
      B[i][j] = A[P[i]][j];
    }
  }

  return B;
}

double compare_two_array(double** A, double** B, int matrix_size){
  double diff = 0.0;

  #pragma omp parallel shared(diff)
  {
    #pragma omp for reduction(+:diff)
    for (int i = 0; i < matrix_size; i++){
      for (int j = 0; j < matrix_size; j++){
        diff += abs(A[i][j] - B[i][j]);
      }
    }
  }

  return diff / matrix_size;
}

double** init_A(int matrix_size){
  double **A = new double*[matrix_size];

  struct drand48_data randBuffer;
  
  #pragma omp parallel shared(A) private(randBuffer)
  {
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

  #pragma omp parallel shared(P)
  {
    #pragma omp for
    for (int i = 0; i < matrix_size; i++){
      P[i] = i;
    }
  }

  return P;
}

double ** init_diagonal_array(int matrix_size){
  double **A = new double*[matrix_size];

  #pragma omp parallel shared(A)
  {
    #pragma omp for  
    for (int i = 0; i < matrix_size; i++){
      A[i] = new double[matrix_size];
      for (int j = 0; j < matrix_size; j++){
        if(i==j) {
          A[i][j] = 1.0;
        } else {
          A[i][j] = 0; 
        }
      }
    }
  }

  return A;
}

void lu_decomposition(double **A, int *P, int matrix_size) {
  for (int k = 0; k < matrix_size; k++) {
    // Pivoting
    double max = 0.0;
    int k_prime = k;

    // Find the maximum element in the k-th column
    #pragma omp parallel shared(max, k_prime)
    { 
      // Initialize local variables for each thread
      double local_max = 0.0;
      int local_k_prime = k;

      #pragma omp for
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

    #pragma omp parallel for
    for (int i = k + 1; i < matrix_size; i++) {
      A[i][k] /= pivot;
      double L_ik = A[i][k];
      // U의 역할을 하는 A의 상삼각 부분을 갱신합니다.
      for (int j = k + 1; j < matrix_size; j++) {
          A[i][j] -= L_ik * A[k][j];
      }
    }

  }
}

void decompose_A_to_L_U(double **A, double **L, double **U, int matrix_size) {
  #pragma omp parallel for
  for (int i = 0; i < matrix_size; i++) {
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