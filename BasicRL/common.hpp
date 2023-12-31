#pragma once	// Avoid recurive #include issues

#include "stdafx.hpp"


// Print an error message, wait for the user to press enter, then exit with error code 1.
[[noreturn]] void errorExit(const std::string& s);

// Returns one index that achieves the maximum value in v. If there are ties, a winner is selected uniformly randomly
int maxIndex(const Eigen::VectorXd& v, std::mt19937_64& generator);

// Returns a vector containing the indices with maximum values.
std::vector<int> maxIndices(const Eigen::VectorXd& v);

// Loads buff[i] with e^x[i] / (sum_j e^x[j]). Useful for softmax, uses the log-sum-exp trick for numerical stability
// Based on: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
void softmax(const Eigen::VectorXd& x, Eigen::VectorXd& buff);

// Same as softmax above, but implemented the naive way that is easier to check. This is used for making sure the above implementation is correct
void softmaxDebug(const Eigen::VectorXd& x, Eigen::VectorXd& buff);

// Return random integer based on the provided probabilities
template <typename VectorType>
int randp(const VectorType & probabilities, std::mt19937_64 & generator)
{
	if (probabilities.size() == 0)
		errorExit("Error in randp - the provided array is length zero.");
	
	std::uniform_real_distribution<double> d(0, 1);
	double sample = d(generator), sum = 0;
	int n = (int)probabilities.size();
	for (int i = 0; i < n; i++)
	{
		sum += probabilities[i];
		if (sample <= sum)
			return i;
	}
	// Let's return the first item with non-zero probability
	for (int i = 0; i < n; i++)
		if (probabilities[i] > 0)
			return i;

	assert(false);	// All probabilities were zero - that shouldn't have happened
	errorExit("Error in randp - all elements have zero probability");
}

/*
Floating-point modulo:
The result (the remainder) has same sign as the divisor.
Similar to matlab's mod(); Not similar to fmod() because:
Mod(-3,4)= 1
fmod(-3,4)= -3
*/
template <typename Scalar>
Scalar Mod(const Scalar& x, const Scalar& y) {
	if (0. == y) return x;
	Scalar m = x - y * (Scalar)std::floor(x / y);
	// handle boundary cases resulted from floating-point cut off:
	if (y > 0) {
		if (m >= y)
			return 0;
		if (m < 0) {
			if (y + m == y) return 0;
			else return (y + m);
		}
	}
	else
	{
		if (m <= y) return 0;
		if (m > 0) {
			if (y + m == y) return 0;
			else return (y + m);
		}
	}
	return m;
}

// wrap [rad] angle to [-PI..PI)
template <typename Scalar>
Scalar wrapPosNegPI(const Scalar& theta) {
	return Mod((Scalar)theta + M_PI, (Scalar)2.0 * M_PI) - (Scalar)M_PI;
}

// wrap [rad] angle to [0..TWO_PI)
template <typename Scalar>
Scalar wrapTwoPI(const Scalar& theta) {
	return Mod((Scalar)theta, (Scalar)(2.0 * M_PI));
}

// wrap [deg] angle to [-180..180)
template <typename Scalar>
Scalar wrapPosNeg180(const Scalar& theta) {
	return Mod((Scalar)(theta + 180.0), (Scalar)360.0) - (Scalar)180.0;
}

// wrap [deg] angle to [0..360)
template <typename Scalar>
Scalar wrap360(const Scalar& theta) {
	return Mod((Scalar)theta, (Scalar)360.0);
}

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

// Function to compute the sample mean of a vector. This works for vector<number type> or for VectorXd (via templating)
template <typename T>
double sampleMean(const T& v)
{
	double result = 0;
	for (int i = 0; i < (int)v.size(); i++)
		result += v[i];
	result /= (double)v.size();
	return result;
}

// Function to compute the sample standard error of a vector. This works for vector<number type> or for VectorXd (via templating)
template <typename T>
double sampleStandardError(const T& v)
{
	if (v.size() < 2)
		return 0;			// Avoid division by zero if v.size() == 1. In that case the standard error is zero

	double result = 0, mu = sampleMean(v);
	for (int i = 0; i < (int)v.size(); i++)
		result += (v[i] - mu) * (v[i] - mu);
	result /= (double)(v.size() - 1);
	// We now have the sample variance. Take the square root to get sample standard deviation
	result = sqrt(result);
	// Next, divide by sqrt(n) to get the sample standard error, where n is the length of the vector
	result = result / sqrt((double)v.size());
	// Return the computed result
	return result;
}

// Pushes back a provided element n times
template <typename T>
void push_back_n(const T& value, const int n, std::vector<T>& buff)
{
	for (int i = 0; i < n; i++)
		buff.push_back(value);
}

// Pushes back n elements from 0 to n 
void push_back_0_n(const int n, std::vector<int>& buff);

// Samples hyper parameters around pre-defined values: :)
// Alpha: samples from range [0.000001, 0.1] around values 0.1, 0.01, 0.001, etc. (with some variation)
// Beta: samples from range [0.000001, 0.1] around values 0.1, 0.01, 0.001, etc. (with some variation)
// Epsilon: samples from range [0.001, 0.1] around values 0.1, 0.05, 0.01, 0.005, 0.001 (with some variation)
// Lambda: samples from range (0,1] around values 0.1, 0.2, 0.3, etc. (with some variation)

double sampleHyperParameter(const std::string HyperParamName, std::mt19937_64& gen);

double sampleParameter(const std::string HyperParamName, std::mt19937_64& gen);

// Reads an CSV file and returns a matrix
std::vector<std::vector<double>> readCSVToMatrix(const std::string& filename);

// Converts a 2D matrix to a 1D vector if possible
std::vector<double> convertTo1D(const std::vector<std::vector<double>>& matrix);