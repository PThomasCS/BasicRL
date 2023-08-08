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
