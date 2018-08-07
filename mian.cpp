#include < iostream > 
#include < mpi.h > 
#include < stdlib.h >

using namespace std;

int compareArray(double * x, double * new_x, int size) {

  for (register int i = 0; i < size; i++) {
    //0.000001 
    
    if ((x[i] - new_x[i]) >= 0.001) {
      return 1;
    }
    
  }
  
  return 0;
}

int main(int argc, char * argv[]) {
  const int n = 200;
  int myid, numprocs;
  char name[MPI_MAX_PROCESSOR_NAME + 1];
  int length;

  MPI_Init( & argc, & argv);
  MPI_Comm_size(MPI_COMM_WORLD, & numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, & myid);
  MPI_Get_processor_name(name, & length);

  if (n % numprocs != 0) {

    if (myid == 0) { //prints only once 
      std::cout << "number of proccess should be divisible by " << n << endl;
    }

  } else {

    int rowSize = n / numprocs; //2 
    int accurate = 1;
    double m[n][n];
    double b[n];
    int start, end;
    double x[n], prev_x[n];
    double new_x[rowSize];
    double b_scatter[rowSize], m_scatter[rowSize * n];

    start = myid * rowSize;
    end = start + rowSize - 1;

    if (myid == 0) {
      std::cout << "numprocs = " << numprocs << " n = " << n << " row size = " << rowSize << end;

      register int r, c, d;

      for (r = 0; r < n; r++)
        for (c = 0; c < n; c++) {

          d = abs(r - c);

          if (d == 0) m[r][c] = -2.0;
          if (d == 1) m[r][c] = 1.0;
          if (d > 1) m[r][c] = 0.0;

        }

      b[0] = -1.0 * (double)(n + 1);

      for (r = 1; r < n; r++) b[r] = 0.0;

      for (register int i = 0; i < n; i++) x[i] = 1.0;

    }

    std::cout << "proccess: " << myid << " proccessor: " << name << " section: " << start << " to " << end << std::endl;

    MPI_Scatter( & m[0][0], rowSize * n, MPI_DOUBLE, m_scatter, rowSize * n, MPI_DOUBLE, 0, MPI_COMM_WORLD); //share sections of m to each proccess 

    MPI_Scatter( & b[0], rowSize, MPI_DOUBLE, b_scatter, rowSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); //send b 

    MPI_Bcast( & x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD); //send current guess to all proccesses 

    do {

      if (myid == 0) {

        for (register int i = 0; i < n; i++) {
          prev_x[i] = x[i]; //store previous guesses 
        }

      }

      int row = 0;
      int pos;

      for (register int i = start; i <= end; i++) { //each section 

        double sum = 0.0;

        for (register int j = 0; j < n; j++) {

          if (i != j) {

            pos = (row * n) + j; //[row][j] 
            sum += m_scatter[pos] * x[j];

          }

        }

        pos = (row * n) + i; //[row][i] 
        new_x[row] = (b_scatter[row] - sum) / m_scatter[pos];
        row++;

      }

      MPI_Allgather( & new_x, rowSize, MPI_DOUBLE, x, rowSize, MPI_DOUBLE, MPI_COMM_WORLD); //accumulate new guesses 

      if (myid == 0) {
        accurate = compareArray(prev_x, x, n);
      }

      MPI_Bcast( & accurate, 1, MPI_INT, 0, MPI_COMM_WORLD);

    } while (accurate == 1);

    if (myid == 0) {

      for (register int j = 0; j < n; j++) {
        std::cout << "result[" << j << "] = " << x[j] << endl;
      }

    }

  }

  MPI_Finalize();
  return 0;
}
