// thrust library
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
using namespace std;

const unsigned int N = 3;
const unsigned int GENERATION_SIZE = 100;
const unsigned int start_xd_min[N] = { 2112,1180,417 };
unsigned int individuals[N][GENERATION_SIZE] = {};

#include "cpu_rnd.cuh"
#include "cuda_rnd.cuh"

#include "compute_time_linux.cuh"

const int n = 3;
const int ns = 2;
const int allNodes = n + ns;
int LT[allNodes][n] = {};

const int x_inf = 100000;
const int simTime = 60;

void gpu()
{
	// device storage for doubles
	thrust::device_vector<double> population(N * GENERATION_SIZE);

	// Fill the vector with random distribution (0,1)
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::transform(index_sequence_begin,
		index_sequence_begin + GENERATION_SIZE * N,
		population.begin(),
		cuda_rnd());


	thrust::device_vector<int> LTcuda(allNodes * n);

	thrust::copy(&(LT[0][0]), &(LT[allNodes - 1][n - 1]), LTcuda.begin());
	thrust::device_vector<int>::iterator iter =
		thrust::max_element(LTcuda.begin(), LTcuda.end());
	int Lcuda = *iter;

	ComputeTimeStart();
	for (int node = 0; node < N; node++) {
		for (int individualNo = 0; individualNo < GENERATION_SIZE; individualNo++) {
			int index = individualNo + node * GENERATION_SIZE;
			population[index] = round(population[index] * start_xd_min[node]);
		}
	}

	// print contents of population
	for (int i = 0; i < population.size(); i++) {
		if (i % GENERATION_SIZE == 0) {
			cout << "\n\nNODE\n";
		}
		cout << population[i] << " ";
	}

	double tt = ComputeTimeEnd();
	cout << "\nGPU time: " << tt << " ms\n";
}

#include "vector_types.cuh"
#include "vector_operators.cuh"
#include "vector_operations.cuh"
#include "eigen_operations.cuh"

void cpu()
{
	ComputeTimeStart();
	for (int node = 0; node < N; node++) {
		for (int individualNo = 0; individualNo < GENERATION_SIZE; individualNo++) {
			individuals[node][individualNo] = static_cast<int>(round(rnd() * start_xd_min[node]));
		}
	}

	for (int node = 0; node < N; node++) {
		cout << "\nNODE\n";
		for (int individualNo = 0; individualNo < GENERATION_SIZE; individualNo++) {
			cout << individuals[node][individualNo] << " ";
		}
		cout << "\n";
	}

	LT[3][0] = 2;
	LT[4][1] = 4;
	LT[1][2] = 3;
	LT[0][2] = 3;
	LT[0][1] = 1;

	int* start = &LT[0][0];
	// max lead time
	int L = *max_element(start, start + allNodes * n);

	float LA_nom[allNodes][n] = {};
	three_dimension_vector_float LA(simTime, vector<vector<float>>(allNodes, vector<float>(n)));

	LA[0][3][0] = LA_nom[3][0] = 1;
	LA[0][4][1] = LA_nom[4][1] = 0.8f;
	LA[0][1][2] = LA_nom[1][2] = 0.6f;
	LA[0][0][2] = LA_nom[0][2] = 0.4f;
	LA[0][0][1] = LA_nom[0][1] = 0.2f;
	//Print2DVector<float>(LA[0], "delay matrix");

	// Verify if allocation correct - elements in each column should sum up to 1 or 0	
	for (int j = 0; j < n; j++) {
		float temp = 0;
		for (int i = 0; i < allNodes; i++) {
			temp = temp + LA[0][i][j];
		}

		if (temp != 0 && temp != 1) {
			throw ("Improper allocation in column: %d", j);
		}
	}

	// Initial conditions
	int time[simTime] = {};
	int u[n][simTime] = {};
	int u_hist[n][simTime] = {}; // order history
	two_dimension_vector_int x(simTime, vector<int>(allNodes, 0));
	int y[allNodes][simTime] = {};
	int xd[allNodes] = { 80, 120, 98, 0, 0 };

	x[0] = { 70, 140, 88, x_inf, x_inf }; // initial stock level
	//Print2DVector<int>(x, "stock level ");

	// Demand
	vector<int> dmax{ 10, 15, 20 };
	two_dimension_vector_int d(n, vector<int>(simTime, {})); // int d[n][simTime] = {};

	for (int j = 0; j < simTime; j++) {
		double multiplier = 1;
		for (int k = 0; k < n; k++) {
			double demand = multiplier * dmax[k] * rnd();
			int randomDemand = static_cast<int>(round(demand));
			if (randomDemand > dmax[k]) d[k][j] = dmax[k];
			else d[k][j] = randomDemand;
		}
	};
	//Print2DVector<int>(d, "demand");

	// State - space description
	// System matrices
	three_dimension_vector_float B_nom(L, two_dimension_vector_float(n, vector<float>(n)));
	four_dimension_vector_float B(simTime, three_dimension_vector_float(L, two_dimension_vector_float(n, vector<float>(n))));

	// Assuming zero order processing time(eq 9)
	two_dimension_vector_float B_0(n, vector<float>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			B_0[i][j] = LA[0][i][j] * -1;
		}
	}

	// index k corresponds to delay k(eq 8)	
	for (int k = 0; k < L; k++) {
		for (int j = 0; j < n; j++) {
			float t_sum = 0;
			for (int i = 0; i < allNodes; i++) {
				if (LT[i][j] == k + 1) {
					t_sum = t_sum + LA[0][i][j];
				}
			}
			B_nom[k][j][j] = t_sum;
		}
	}
	//Print3DVector<float>(B_nom, "B matrix");
	B[0] = B_nom;

	//% Sum of delay matrices
	two_dimension_vector_float Lambda(n, vector<float>(n, {}));

	// table index k corresponds to delay k
	for (int k = 0; k < L; k++) {
		Lambda = Lambda + B[0][k];
	}

	Lambda = Lambda + B_0; // eq 11 

	vector<float> xd_min = GetXdMin(n, L, B, Lambda, dmax);
	PrintVector(xd_min, "xd min");

	double tt = ComputeTimeEnd();
	cout << "CPU time: " << tt << " ms\n";
}

int main()
{
	cpu();
	gpu();

	return 0;
}