// thrust library
#include <thrust/device_vector.h>

using namespace std;

const unsigned int N = 3;
const unsigned int GENERATION_SIZE = 100;
const unsigned int start_xd_min[N] = { 2112,1180,417 };
unsigned int individuals[N][GENERATION_SIZE] = {};

#include "cpu_rnd.cuh"
#include "cuda_rnd.cuh"

#include "compute_time.cuh"

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

	double tt = ComputeTimeEnd();
	cout << "CPU time: " << tt << " ms\n";
}

int main()
{
	cpu();
	gpu();

	return 0;
}