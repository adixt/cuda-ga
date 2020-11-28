#pragma once
#include "cuda_runtime.h"

// thrust library
#include <thrust/random.h>

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