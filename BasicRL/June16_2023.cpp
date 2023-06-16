#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

// Return the probability density of x, for the normal with mean mu and variance sigmaSquared
double normalPDF(double x, double mu, double sigmaSquared)
{
	return 1.0 / (2.0 * M_PI * sigmaSquared) * exp(-0.5*(x - mu)* (x - mu) / sigmaSquared);
}

// Experiments with basic importance sampling
void sandboxJune16_2023()
{
	mt19937_64 generator;

	double pMean = 0, qMean = 3, pSigmaSquared = 1, qSigmaSquared = 1;
	normal_distribution<double> p(pMean, pSigmaSquared), q(qMean, qSigmaSquared);

	int n = 500000;

	// Draw samples
	VectorXd samples(n);
	for (int i = 0; i < n; i++)
		samples[i] = q(generator);
	
	// Compute the IS estimate of the mean of p
	double ISEstimate = 0;
	for (int i = 0; i < n; i++)
	{
		ISEstimate += normalPDF(samples[i], pMean, pSigmaSquared) / normalPDF(samples[i], qMean, qSigmaSquared) * samples[i];
	}
	ISEstimate /= (double)n;	// (double) is making sure that we treat n as a floating point number, not an integer

	// What if we could sample from p? Let's draw n samples and compare.
	VectorXd samplesFromP(n);
	for (int i = 0; i < n; i++)
		samplesFromP[i] = p(generator);
	double meanFromPSamples = samplesFromP.mean();


	// Print the result
	cout << "pMean \t\t\t\t\t\t= " << pMean << endl;
	cout << "Estimate of pMean by sampling from p \t\t= " << meanFromPSamples << endl;
	cout << "IS estimate of pMean when sampling from q \t= " << ISEstimate << endl;
	
}