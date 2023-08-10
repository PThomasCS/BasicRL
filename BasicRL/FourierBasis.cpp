#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

FourierBasis::FourierBasis(int inputDimension, const Eigen::VectorXd& inputLowerBound, const Eigen::VectorXd& inputUpperBound, int iOrder, int dOrder)
{
	// Copy over the provided arguments
	this->inputDimension = inputDimension;
	this->inputLowerBound = inputLowerBound;
	this->inputUpperBound = inputUpperBound;
	this->iOrder = iOrder;
	this->dOrder = dOrder;

	// initialize maxMinusMinValues
	inputRange = (inputUpperBound - inputLowerBound);

	// If both orders are zero, don't compute anything else - we will just pass through the input
	if ((iOrder == 0) && (dOrder == 0))
	{
		numOutputs = inputDimension;
		return;
	}
	
	// Compute the total number of terms
	int iTerms = iOrder * inputDimension;						// Number of independent terms
	int dTerms = ipow(dOrder + 1, inputDimension);				// Number of dependent terms
	int oTerms = min(iOrder, dOrder) * inputDimension;			// Overlap of iTerms and dTerms
	numOutputs = iTerms + dTerms - oTerms;

	// Initialize c
	if (inputDimension <= 0)
		errorExit("FourierBasis::FourierBasis requires at least one input dimension.");
	c.resize(numOutputs, inputDimension);
	VectorXi counter(inputDimension);
	// First add the dependent terms
	counter.setZero();							        // Set all terms to zero in the counter (which counts in base dOrder
	int termCount;
	for (termCount = 0; termCount < dTerms; termCount++)
	{
		for (int i = 0; i < inputDimension; i++)
			c(termCount, i) = (float)counter(i);
		incrementCounter(counter, dOrder);
	}
	// Add the independent terms
	for (int i = 0; i < inputDimension; i++)
	{
		for (int j = dOrder + 1; j <= iOrder; j++)
		{
			c.row(termCount).setZero();
			c(termCount, i) = (float)j;
			termCount++;
		}
	}
}

string FourierBasis::getName() const
{
	return "Fourier Basis with iOrder = " + to_string(iOrder) + ", dOrder = " + to_string(dOrder);
}

void FourierBasis::generateFeatures(const Eigen::VectorXd& in, Eigen::VectorXd& outBuff)
{
	// Normalize the state
	normalizedInput = (in - inputLowerBound).array() / inputRange;

	// If both orders are zero, just pass through the normalized input
	if ((iOrder == 0) && (dOrder == 0))
	{
		outBuff = normalizedInput;
		return;
	}

	// Compute the features
	outBuff = (M_PI * c * normalizedInput).array().cos();
}

int FourierBasis::getNumOutputs() const
{
	return numOutputs;
}
