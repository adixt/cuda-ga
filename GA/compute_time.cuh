#pragma once
#include <Windows.h>

using namespace std;

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