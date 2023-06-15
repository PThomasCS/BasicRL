#pragma once	// Avoid recurive #include issues

#include "stdafx.hpp"


// Print an error message, wait for the user to press enter, then exit with error code 1.
void errorExit(const std::string& s);

int maxIndex(const Eigen::VectorXd& v, std::mt19937_64& generator);

// Just like the "pow" function (power), but for integers.
// See this link for details: https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c
template <typename intType>
intType ipow(const intType& a, const intType& b)
{
	if (b == 0) return 1;
	if (b == 1) return a;
	intType tmp = ipow(a, b / 2);
	if (b % 2 == 0) return tmp * tmp;
	else return a * tmp * tmp;
}

// Treat counter as a vector containing digits (actually integers). The number
// represented by the digits is incremented (zeroth digit is incremented first)
template <typename IntVectorType>
void incrementCounter(IntVectorType& counter, int maxDigit)
{
	for (int i = 0; i < (int)counter.size(); i++)
	{
		counter[i]++;
		if (counter[i] > maxDigit)
			counter[i] = 0;
		else
			break;
	}
}