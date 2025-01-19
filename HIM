1111111111111111
#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

int main() {
	int n;
	cout << "Enter matrix size: ";
	cin >> n;
	vector<vector<int>> A(n, vector<int>(n));
	vector<int> x(n), y_serial(n, 0), y_parallel(n, 0);

	cout << "Enter matrix A :\n";
	for (auto &row : A) for (int &a : row) cin >> a;
	cout << "Enter vector x: \n";
	for (int &xi : x) cin >> xi;

	// Serial execution
	double start_serial = omp_get_wtime();
	for (int i = 0; i < n; i++)
    	for (int j = 0; j < n; j++)
        	y_serial[i] += A[i][j] * x[j];
	double end_serial = omp_get_wtime();

	// Parallel execution
	double start_parallel = omp_get_wtime();
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
    	for (int j = 0; j < n; j++)
        	y_parallel[i] += A[i][j] * x[j];
	double end_parallel = omp_get_wtime();

	cout << "Serial Result y = ";
	for (int yi : y_serial) cout << yi << " ";
	cout << "\nSerial Time: " << end_serial - start_serial << " seconds\n";

	cout << "Parallel Result y = ";
	for (int yi : y_parallel) cout << yi << " ";
	cout << "\nParallel Time: " << end_parallel - start_parallel << " seconds\n";
    
	return 0;
}

OUPUT:
gedit prog.cpp 
g++ -fopenmp prog.cpp
./a.out

Enter matrix size: 3
Enter matrix A :
1 2 3
4 5 6
7 8 9
Enter vector x:
1 2 3
Serial Result y = 14 32 50
Serial Time: 9.91e-07 seconds
Parallel Result y = 14 32 50
Parallel Time: 0.00141599 seconds

22222222222

#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
	vector<string> sections = {"Clothing", "Gaming", "Grocery", "Stationery"};
	vector<int> prices_parallel(sections.size(), 0), prices_serial(sections.size(), 0);

	// Serial execution
	cout<<"Serial execution: \n";
	double start_serial = omp_get_wtime();
	for (int i = 0; i < sections.size(); ++i) {
    	int num_items, total = 0;
    	cout << "Enter items & prices for " << sections[i] << " (Serial):\n";
    	cin >> num_items;
    	for (int j = 0; j < num_items; ++j) {
        	int price;
        	cin >> price;
        	total += price;
    	}
    	prices_serial[i] = total;
	}
	double end_serial = omp_get_wtime();

	// Parallel execution
	cout<<"\nParallel execution:\n";
	double start_parallel = omp_get_wtime();
	for (int i = 0; i < sections.size(); ++i) {
    	int num_items, total = 0;
    	cout << "Enter items & prices for " << sections[i] << " (Parallel):\n";
    	cin >> num_items;

    	#pragma omp parallel for reduction(+:total)
    	for (int j = 0; j < num_items; ++j) {
        	int price;
        	cin >> price;
        	total += price;
    	}
    	prices_parallel[i] = total;
	}
	double end_parallel = omp_get_wtime();

	// Final summary
	cout << "\nSerial Prices:\n";
	int overall_serial = 0;
	for (int i = 0; i < sections.size(); ++i) {
    	cout << sections[i] << ": " << prices_serial[i] << "\n";
    	overall_serial += prices_serial[i];
	}
	cout << "Overall Cost (Serial): " << overall_serial << "\n";
	cout << "Serial Time: " << end_serial - start_serial << " seconds\n";

	cout << "\nParallel Prices:\n";
	int overall_parallel = 0;
	for (int i = 0; i < sections.size(); ++i) {
    	cout << sections[i] << ": " << prices_parallel[i] << "\n";
    	overall_parallel += prices_parallel[i];
	}
	cout << "Overall Cost (Parallel): " << overall_parallel << "\n";
	cout << "Parallel Time: " << end_parallel - start_parallel << " seconds\n";

	return 0;
}


OUPUT:
gedit prog.cpp 
g++ -fopenmp prog.cpp
./a.out

Serial execution:
Enter items & prices for Clothing (Serial):
3
500 600 700
Enter items & prices for Gaming (Serial):
2
1500 2500
Enter items & prices for Grocery (Serial):
4
100 200 300 400
Enter items & prices for Stationery (Serial):
3
50 60 70

Parallel execution:
Enter items & prices for Clothing (Parallel):
3
500 600 700
Enter items & prices for Gaming (Parallel):
2
1500 2500
Enter items & prices for Grocery (Parallel):
4
100 200 300 400
Enter items & prices for Stationery (Parallel):
3
50 60 70

Serial Prices:
Clothing: 1800
Gaming: 4000
Grocery: 1000
Stationery: 180
Overall Cost (Serial): 6980
Serial Time: 19.3207 seconds

Parallel Prices:
Clothing: 1800
Gaming: 4000
Grocery: 820
Stationery: 180
Overall Cost (Parallel): 6800
Parallel Time: 19.8164 seconds



3333333333333333
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>
void main() {
int num,i;
printf("Enter the number of steps : ");
scanf("%d",&num);
time_t st,et;
st=clock();
double step=1.0/(double)num,pi=0.0;
omp_set_num_threads(num);
#pragma omp parallel for reduction(+:pi)
for(i=0;i<num;i++) {
double x=(i+0.5)*step;
double local_pi=(4.0*step)/(1+x*x);
pi+=local_pi;
}
et=clock();
printf("Time Taken : %lf\n",(double)((double)(et-st)/CLOCKS_PER_SEC));
printf("Value of Pi = %.16lf\n",pi);
}

//gcc p3.c -fopenmp
./a.out
enter the no.of steps:100



44444444444444444
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>

void main() {

int n, i;

printf("Number of terms : ");
scanf("%d",&n);

int* a = (int*)malloc(n * sizeof(int));
a[0] = 0;
a[1] = 1;

time_t st, et;
st = clock();

omp_set_num_threads(2);

#pragma omp parallel
{
#pragma omp single
{
printf("id of thread involved in the computation of fibonacci numbers = %d\n", omp_get_thread_num());
for (i = 2; i < n; i++)
a[i] = a[i - 2] + a[i - 1];
}
#pragma omp single
{
printf("id of thread involved in the displaying of fibonacci numbers = %d\n", omp_get_thread_num());
printf("Fibonacci numbers : ");
for (i = 0; i < n; i++)
printf("%d ", a[i]);
printf("\n");
}
}
et = clock();
printf("Time Taken : %lfms\n", ((double)(et - st)*1000 / CLOCKS_PER_SEC));
}

//gcc p4.c -fopenmp
//    ./a.out
No.of terms: 16


555555555555555
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>
void main() {
int n, i;
time_t st, et;
st = clock();
printf("Enter the number of students : ");
scanf("%d", &n);
double* arr = (double*)malloc(n * sizeof(double));
double arr_max = 0;
#pragma omp parallel for
for (i = 0; i < n; i++) {
srand(i);
arr[i] = (double)(rand() % 10000)/10 ;
}
printf("CGPA of students : ");
for (i = 0; i < n; i++)
printf("%.2lf ", arr[i]);
printf("\n");
#pragma omp parallel for
for (i = 0; i < n; i++) {
#pragma omp critical
if (arr_max < arr[i])
arr_max = arr[i];
}
et = clock();
printf("Student with highest CGPA = %.2lf\n", arr_max);
printf("Time Taken : %.2lfms\n", ((double)(et - st) * 1000 / CLOCKS_PER_SEC));
}
gcc p.c -fopenmp
./a.out
or
#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

int main() {
	int num_students;
	cout << "Enter number of students: ";
	cin >> num_students;

	vector<double> CGPA(num_students);
	cout << "Enter the CGPAs of the students:\n";
	for (double &cgpa : CGPA) cin >> cgpa;

	double max_cgpa_serial = CGPA[0], max_cgpa_parallel = CGPA[0];

	// Serial execution
	double start_serial = omp_get_wtime();
	for (int i = 0; i < num_students; i++)
    	if (CGPA[i] > max_cgpa_serial) max_cgpa_serial = CGPA[i];
	double end_serial = omp_get_wtime();

	// Parallel execution
	double start_parallel = omp_get_wtime();
	#pragma omp parallel for shared(CGPA, max_cgpa_parallel)
	for (int i = 0; i < num_students; i++) {
    	#pragma omp critical
    	{
        	if (CGPA[i] > max_cgpa_parallel)
            	max_cgpa_parallel = CGPA[i];
    	}
	}
	double end_parallel = omp_get_wtime();

	cout << "Highest CGPA (Serial): " << max_cgpa_serial << "\n";
	cout << "Serial Time: " << end_serial - start_serial << " seconds\n";

	cout << "Highest CGPA (Parallel): " << max_cgpa_parallel << "\n";
	cout << "Parallel Time: " << end_parallel - start_parallel << " seconds\n";

	return 0;
}


OUTPUT:
gedit prog.cpp 
g++ -fopenmp prog.cpp
./a.out
Enter number of students: 5
Enter the CGPAs of the students:
9.5
8.6
9.7
9.3
8.8
Highest CGPA (Serial): 9.7
Serial Time: 7.67e-07 seconds
Highest CGPA (Parallel): 9.7
Parallel Time: 0.000998799 seconds

gedit prog.cpp 
g++ -fopenmp prog.cpp
./a.out



666666666666666
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
int main(int argc, char** argv) {
int rank, numproc;
int sum = 0;
int total_sum = 0;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &numproc);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
srand(rank);
sum = rand() % 100;
printf("Robot %d picked %d mangoes.\n", rank, sum);
// Start timing
double start_time = MPI_Wtime();
MPI_Reduce(&sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
// End timing
double end_time = MPI_Wtime();
if (rank == 0) {
printf("Total Mangoes picked by %d Robots = %d\n", numproc, total_sum);
printf("Time taken for the computation: %f seconds\n", end_time - start_time);
}
MPI_Finalize();
return 0;
}
mpicc -o p p.c
mpirun -np 8 ./p


777777777777777
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
int main(int argc, char* argv[])
{
int size, rank;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
float recvbuf, sendbuf[100];
double start_time, end_time, time_taken;
// Record start time
start_time = MPI_Wtime();
if (rank == 0) {
int i;
printf("Before Scatter : sendbuf of rank 0 : ");
for (i = 0; i < size; i++) {
srand(i); // seeding random number generator for different values on each process
sendbuf[i] = (float)(rand() % 1000) / 10;
printf("%.1f ", sendbuf[i]);
}
printf("\nAfter Scatter :\n");
}
// Perform the scatter operation
MPI_Scatter(sendbuf, 1, MPI_FLOAT, &recvbuf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
// Print the received data
printf("rank= %d Recvbuf: %.1f\n", rank, recvbuf);
// Record end time
end_time = MPI_Wtime();
// Calculate time taken for scatter operation
time_taken = end_time - start_time;
// Only rank 0 will print the total time taken
if (rank == 0) {
printf("Time taken for MPI_Scatter: %.6f seconds\n", time_taken);
}
MPI_Finalize();
OUTPUT
}
mpicc -o p p.c
mpirun -np 8 ./p

888888888888
#include <iostream>
#include <mpi.h>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
	
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int dims[2], coords[2], north, south, east, west, value;
    int periods[2] = {0, 0}; 
    MPI_Comm cart_comm;

    MPI_Dims_create(world_size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east); 

    value = world_rank; 
    cout << "Process " << world_rank << " at (" << coords[0] << ", " << coords[1] << ") has value: " << value << endl;

    if (north != MPI_PROC_NULL) MPI_Send(&value, 1, MPI_INT, north, 0, cart_comm);
    if (south != MPI_PROC_NULL) MPI_Recv(&value, 1, MPI_INT, south, 0, cart_comm, MPI_STATUS_IGNORE);
    if (west != MPI_PROC_NULL) MPI_Send(&value, 1, MPI_INT, west, 0, cart_comm);
    if (east != MPI_PROC_NULL) MPI_Recv(&value, 1, MPI_INT, east, 0, cart_comm, MPI_STATUS_IGNORE);

    MPI_Finalize();
    return 0;
}

/*
gedit ak8.cpp
mpic++ ak8.cpp
mpirun -np 4 ./a.out

Process 0 at (0, 0) has value: 0
Process 1 at (0, 1) has value: 1
Process 2 at (1, 0) has value: 2
Proce
*/


999999999
#include <iostream>
#include <mpi.h>
#include <vector>

using namespace std;
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int TAG = 0, DATA_SIZE = 10;
    vector<int> send_data(DATA_SIZE, world_rank); 
    vector<int> recv_data(DATA_SIZE);              

    MPI_Request send_request, recv_request;

    if (world_rank == 0) {
        cout << "Process 0 sending data (blocking): ";
        for (int i : send_data) cout << i << " ";
        cout << endl;

        // Blocking send
        MPI_Send(send_data.data(), DATA_SIZE, MPI_INT, 1, TAG, MPI_COMM_WORLD);
        cout << "Process 0: Blocking send completed." << endl;

        // Non-blocking send
        MPI_Isend(send_data.data(), DATA_SIZE, MPI_INT, 1, TAG, MPI_COMM_WORLD, &send_request);
        cout << "Process 0 non-blocking send initiated." << endl;
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);  
        cout << "Process 0 non-blocking send completed." << endl;
    } 
    else if (world_rank == 1) {
			
        // Blocking receive
        MPI_Recv(recv_data.data(), DATA_SIZE, MPI_INT, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Process 1 received data (blocking): ";
        for (int i : recv_data) cout << i << " ";
        cout << endl;
        cout << "Process 1: Blocking receive completed." << endl;

        // Non-blocking receive
        MPI_Irecv(recv_data.data(), DATA_SIZE, MPI_INT, 0, TAG, MPI_COMM_WORLD, &recv_request);
        cout << "Process 1 non-blocking receive initiated." << endl;
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE); 
        cout << "Process 1 non-blocking receive completed: ";
        for (int i : recv_data) cout << i << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}

/*
mpic++ akhpc9.cpp
mpirun -np 9 ./a.out

Process 0 sending data (blocking): 0 0 0 0 0 0 0 0 0 0 
Process 0: Blocking send completed.
Process 0 non-blocking send initiated.
Process 0 non-blocking send completed.
Process 1 received data (blocking): 0 0 0 0 0 0 0 0 0 0 
Process 1: Blocking receive completed.
Process 1 non-blocking receive initiated.
Process 1 non-blocking receive completed: 0 0 0 0 0 0 0 0 0 0 
*/

1000000000000
#include <iostream>
#include <vector>
#include <omp.h>  // OpenMP
#include <ctime>
 
using namespace std;

void multiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int N, bool parallel) {
    #pragma omp parallel for collapse(2) if(parallel)  // Parallelize if "parallel" is true
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int N;
    cout << "Enter matrix size (e.g., 1000, 2000, 3000): "; cin >> N;

    vector<vector<int>> A(N, vector<int>(N, 1)), B(N, vector<int>(N, 1)), C(N, vector<int>(N, 0));

    clock_t start = clock();
    multiply(A, B, C, N, false);  // Sequential multiplication
    clock_t end = clock();
    cout << "Time taken for sequential multiplication: " << double(end - start) / CLOCKS_PER_SEC << " seconds" << endl;

    start = clock();
    multiply(A, B, C, N, true);   // Parallel multiplication
    end = clock();
    cout << "Time taken for parallel multiplication (OpenMP): " << double(end - start) / CLOCKS_PER_SEC << " seconds" << endl;

    return 0;
}

/*
Enter matrix size (e.g., 1000, 2000, 3000): 1000
Time taken for sequential multiplication: 6.82819 seconds
Time taken for parallel multiplication (OpenMP): 0.62969 seconds
*/

gedit akhpc6.cpp
mpic++ akhpc6.cpp
mpirun -np 5 ./a.out
