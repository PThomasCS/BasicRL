#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

// Print an error message, wait for the user to press enter, then exit with error code 1.
[[noreturn]] void errorExit(const string& s)
{
	cerr << s << endl;
	getchar();
	exit(1);
}

int maxIndex(const VectorXd& v, mt19937_64& generator)
{
	// Check that v has at least one element. This is redundant with the check inside maxIndices below, but won't make a difference in release mode.
	assert((int)v.size() > 0);

	// Initially the first element is best
	vector<int> bestIndices = maxIndices(v);
	
	// When we get here, bestIndices has been loaded. If only one, just return it.
	if ((int)bestIndices.size() == 1)
		return bestIndices[0];
	else
	{
		// There was a tie
		uniform_int_distribution<int> d(0, (int)bestIndices.size() - 1);
		int index = d(generator);
		return bestIndices[index];
	}
}

vector<int> maxIndices(const VectorXd& v)
{
	// Check that v has at least one element.
	assert((int)v.size() > 0);

	// Initially the first element is best
	vector<int> bestIndices(1, 0); // Note: When profiling, this is a surprisingly slow line. If it really matters, you can find ways to not declare this varaible every time this function is called. Be careful about threading though!
	double bestValue = v[0];

	// Loop over the other elements updating our list of bestIndices and the bestValue
	for (int i = 1; i < (int)v.size(); i++)
	{
		if (v[i] == bestValue)
			bestIndices.push_back(i);
		else if (v[i] > bestValue)
		{
			bestIndices.resize(1);
			bestIndices[0] = i;
			bestValue = v[i];
		}
	}

	return bestIndices;
}

// Loads buff[i] with e^x[i] / (sum_j e^x[j]). Useful for softmax, uses the log-sum-exp trick for numerical stability
// Based on: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
void softmax(const Eigen::VectorXd& x, Eigen::VectorXd& buff)
{
	buff.resize(x.size());
	double c = x.maxCoeff();	// c in the linked website, which is the max element of x

	// Compute x-c for use later
	Eigen::VectorXd xMinusC;
	xMinusC = x.array() - c;

	// expOfXMinusC = xMinusC, exponentiated
	Eigen::VectorXd expOfXMinusC = xMinusC.array().exp();

	// Now compute the log of the sum of the elements in temp
	double logSum = log(expOfXMinusC.sum());

	for (int i = 0; i < (int)x.size(); i++)
	{
		// buff[i] = p_i in the linked page
		buff[i] = exp(x[i] - (c + logSum));
	}
}

// Same as softmax above, but implemented the naive way that is easier to check. This is used for making sure the above implementation is correct
void softmaxDebug(const Eigen::VectorXd& x, Eigen::VectorXd& buff)
{
	buff = x.array().exp();
	buff /= buff.sum();
}

// Dummy function for framework test

// Samples hyper parameters around pre-defined values:
// Alpha: samples from range [0.000001, 0.1] around values 0.1, 0.01, 0.001, etc. (with some variation)
// Beta: samples from range [0.000001, 0.1] around values 0.1, 0.01, 0.001, etc. (with some variation)
// Epsilon: samples from range [0.001, 0.1] around values 0.1, 0.05, 0.01, 0.005, 0.001 (with some variation)
// Lambda: samples from range (0,1] around values 0.1, 0.2, 0.3, etc. (with some variation)

// TO-DO: use log-scale? Add the option of deterministically choosing some parameters (sample lambda = 0.8 half of the time) :)
double sampleHyperParameter(const string HyperParamName, std::mt19937_64& generator)
{
	uniform_real_distribution<double> uni_dist;
	vector<double> predefined_values;

	if (HyperParamName == "alpha" || HyperParamName == "beta") {
		predefined_values = { 0.1, 0.01, 0.001, 0.0001, 0.00001 };
		uni_dist = uniform_real_distribution<double>(0.000001, 0.1);
	}
	else if (HyperParamName == "epsilon") {
		predefined_values = { 0.1, 0.05, 0.01, 0.005, 0.001 };
		uni_dist = uniform_real_distribution<double>(0.001, 0.1);
	}
	else if (HyperParamName == "lambda") {
		predefined_values = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
		uni_dist = uniform_real_distribution<double>(0.0, 1.0);
	}

	// Sample a predefined value
	uniform_int_distribution<size_t> index_dist(0, predefined_values.size() - 1);
	double sampledPredefinedValue = predefined_values[index_dist(generator)];

	// Sample an offset (variation) for the predefined value (~10%)
	double variation = uni_dist(generator) * 0.1;
	double sampledHyperParameter = sampledPredefinedValue + (uni_dist(generator) < 0.5 ? variation : -variation);

	// Make sure that the sample value is in range and return the sampled hyper parameter
	return max(uni_dist.a(),min(uni_dist.b(), sampledHyperParameter));
}
