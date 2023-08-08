#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

IdentityBasis::IdentityBasis(int inputDimension, const Eigen::VectorXd& inputLowerBound, const Eigen::VectorXd& inputUpperBound)
{
	// Copy over the provided arguments
	this->inputDimension = inputDimension;
	this->inputLowerBound = inputLowerBound;
	this->inputUpperBound = inputUpperBound;

	// initialize maxMinusMinValues
	inputRange = (inputUpperBound - inputLowerBound);
}

void IdentityBasis::generateFeatures(const Eigen::VectorXd& in, Eigen::VectorXd& outBuff)
{
	// Normalize the state
	outBuff = (in - inputLowerBound).array() / inputRange;
}

int IdentityBasis::getNumOutputs() const
{
	return inputDimension;
}
