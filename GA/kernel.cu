#include <iostream>
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <cmath>
#include <random>
#include <functional>

// thrust library
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

using namespace std;

const unsigned int N = 3;
const unsigned int GENERATION_SIZE = 100;
const int start_xd_min[N] = { 2112,1180,417 };
int individuals[N][GENERATION_SIZE] = {};

// uniform distribution
default_random_engine generator;
uniform_real_distribution<double> distribution(0, 1);
double dice_roll = distribution(generator);
auto rnd = bind(distribution, generator);

/* Generate CUDA uniform distribution
 *
 * Used by thrust transform functions to create large numbers of
 * random numbers in a uniform distribution.
 */
struct cuda_rnd
{
	double a, b;

	__host__ __device__
		cuda_rnd(double _a = 0.f, double _b = 1.f) : a(_a), b(_b) {
	};

	__host__ __device__
		double operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<double> dist(a, b);
		rng.discard(n);

		return dist(rng);
	}
};

LARGE_INTEGER tb, te, tf;

void ComputeTimeStart()
{
	QueryPerformanceFrequency(&tf);
	QueryPerformanceCounter(&tb);
}

double ComputeTimeEnd()
{
	QueryPerformanceCounter(&te);
	return 1000.0 * (double(te.QuadPart - tb.QuadPart)) / double(tf.QuadPart);
}

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