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

// TO-DO: use in the maxIndex function (?)

vector<int> maxIndices(const VectorXd& v, mt19937_64& generator)
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