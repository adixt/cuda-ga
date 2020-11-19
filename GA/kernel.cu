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

#include "compute_time.cuh"

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

void cpu()
{
	ComputeTimeStart();
	for (int node = 0; node < N; node++) {
		for (int individualNo = 0; individualNo < GENERATION_SIZE; individualNo++) {
			individuals[node][individualNo] = round(rnd() * start_xd_min[node]);
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

	double tt = ComputeTimeEnd();
	cout << "CPU time: " << tt << " ms\n";
}

int main()
{
	cpu();
	gpu();

	return 0;
}