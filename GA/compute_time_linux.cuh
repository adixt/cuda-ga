#include <ctime>

using namespace std;

clock_t c_start;
clock_t c_end;

void ComputeTimeStart()
{
	c_start = clock();
}

double ComputeTimeStart()
{
	c_end = clock();
	return 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;;
}