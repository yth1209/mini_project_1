#include <iostream>
#include <omp.h>
#include <time.h>
#include <chrono>
#include <functional>

using namespace std;

void usage(const char *name);
void print_array(double **A, int matrix_size);
void init_A(double **A, int matrix_size);
void measure_time(function<void()> function);
void init_P(int *P, int matrix_size);
void init_diagonal_array(double **A, int matrix_size);

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

  double **A = new double*[matrix_size];
  int *P = new int[matrix_size];
  double **L = new double*[matrix_size];
  double **U = new double*[matrix_size];

  measure_time([&](){init_A(A, matrix_size);});
  measure_time([&](){
    init_P(P, matrix_size);
    init_diagonal_array(L, matrix_size);
    init_diagonal_array(U, matrix_size);
  }); 

  // print_array(A, matrix_size);
  
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
  for(int i = 0; i < matrix_size; i++){
    for(int j = 0; j < matrix_size; j++){
      cout << A[i][j] << " ";
    }
    cout << endl; 
  }
}

void measure_time(function<void()> function){
  auto start = std::chrono::high_resolution_clock::now();

  function();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

void init_A(double **A, int matrix_size){
  #pragma omp parallel shared(A, matrix_size)
  {
    #pragma omp for
    for (int i = 0; i < matrix_size; i++){
      A[i] = new double[matrix_size];
    }

    struct drand48_data randBuffer;
    srand48_r(chrono::high_resolution_clock::now().time_since_epoch().count(), &randBuffer);

    #pragma omp for
    for (int i = 0; i < matrix_size; i++){
      for (int j = 0; j < matrix_size; j++){
        drand48_r(&randBuffer, &A[i][j]);
      }
    }
  }
}

void init_P(int *P, int matrix_size){
  #pragma omp parallel shared(P, matrix_size)
  {
    #pragma omp for
    for (int i = 0; i < matrix_size; i++){
      P[i] = i;
    }
  }
}

void init_diagonal_array(double **A, int matrix_size){
  #pragma omp parallel shared(A, matrix_size)
  {
    #pragma omp for
    for (int i = 0; i < matrix_size; i++){
      A[i] = new double[matrix_size];
    }

    #pragma omp for
    for (int i = 0; i < matrix_size; i++){
      for (int j = 0; j < matrix_size; j++){
        if(i==j) {
          A[i][j] = 1.0;
        } else {
          A[i][j] = 0; 
        }
      }
    }
  }
}