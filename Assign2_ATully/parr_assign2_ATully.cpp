#include <iostream>
#include <chrono>
#include <cmath>
#include <mpi.h>

double exact_solution(double x, double t){                    // function to calculate the exact solution of the heat equation at (x,t)
    double out_val{};
    out_val = exp(-4*pow(M_PI,2)*t) * sin(2*M_PI*x) + 2 * exp(-25*pow(M_PI,2)*t) * sin(5*M_PI*x) + 3 * exp(-400*pow(M_PI,2)*t) * sin(20*M_PI*x);
    return out_val;
}


int main(){
    int M{1010}, N{500000}, j{86}, x_val{};                     // define the space and time parameters along with an indexing parameter
    double T{0.1}, dx{1./M}, dt{T/N}, mu{dt/pow(dx,2)};         // values for the time and space stepping and the courant number
    double U[j][2]{}, U_out[M]{};                               // arrays to store the solutions and the final solution
    int rank{}, size{};                                         // the rank and size variables for the process
    MPI_Comm comm;

    comm = MPI_COMM_WORLD;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int root_process{0};

    auto start = std::chrono::steady_clock::now();          // start the timer for computation time
    

    for (int x{}; x<j; ++x) {
        x_val = rank * (j-2) + x;
        U[x][0] = sin(2*M_PI*x_val*dx) + 2 * sin(5*M_PI*x_val*dx) + 3 * sin(20*M_PI*x_val*dx);      // implement the inital condition for the segment
    }
    if (rank == 0) {                            // enforce boundary conditions
        U[0][0] = 0;
        U[0][1] = 0;
    }
    else if (rank == size -1) {                 // enforce boundary conditions
        U[j-1][0] = 0;
        U[j-1][1] = 0;
    }

    for (int t{1}; t<=N; ++t) {             // main loop of forward Euler
        for (int x{1}; x<j-1; ++x) {
            U[x][1] = U[x][0] + mu * (U[x-1][0] -2 * U[x][0] + U[x+1][0]);
        }
        for (int x{1}; x<j-1; ++x) {
            U[x][0] = U[x][1];              // update the new time step to be the old time step
        }
        
        
        if (rank == 0 ) {                                                                       // boundary cases to send and receive
            MPI_Send(&U[j-2][0], 1, MPI_DOUBLE, (rank+1), 0, comm);
            MPI_Recv(&U[j-1][0], 1, MPI_DOUBLE, (rank+1), 1, comm, MPI_STATUS_IGNORE);
        }
        else if (rank == size-1) {
            MPI_Send(&U[1][0], 1, MPI_DOUBLE, (rank-1), 1, comm);                               // boundary cases to send and receive
            MPI_Recv(&U[0][0], 1, MPI_DOUBLE, (rank-1), 0, comm, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Send(&U[1][0], 1, MPI_DOUBLE, (rank-1), 1, comm);
            MPI_Recv(&U[0][0], 1, MPI_DOUBLE, (rank-1), 0, comm, MPI_STATUS_IGNORE);            // Halo-swapping of all segments except boundary cases above

            MPI_Send(&U[j-2][0], 1, MPI_DOUBLE, (rank+1), 0, comm);
            MPI_Recv(&U[j-1][0], 1, MPI_DOUBLE, (rank+1), 1, comm, MPI_STATUS_IGNORE);
        }
    }

    double U_send[j]{};
    for (int i{}; i<j;++i) {
        U_send[i] = U[i][0];                    // variable to store the final time step to send -- 2D array wasnt sending well
    }   
  
    MPI_Gather(&U_send[1], j-2, MPI_DOUBLE, &U_out[1], j-2, MPI_DOUBLE, 0, comm); // gather all the segments in the root process
    
    if (rank == 0) {
        double ex_sol[M]{};             // compute the exact solution
        for (int x{1}; x<M-1; ++x) {
            ex_sol[x] = exact_solution(x*dx,T);
        }

        std::cout << "Calculated solution (left) and exact solution (right) at T = " << T << ":" << std::endl;
        for (int x{}; x<M; ++x) {
            std::cout << U_out[x] << "    " << ex_sol[x] << std::endl;          // print out the solutions
        }
        std::cout <<  std::endl;
    }
    if (rank == 0) {
        auto end = std::chrono::steady_clock::now();
        auto time_taken = end - start;                      // calculate the final time for computation and print time taken
        std::cout << "Computation time taken: " << std::chrono::duration <double, std::milli> (time_taken).count() << " ms" << std::endl;
    }
}
