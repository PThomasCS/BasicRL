#pragma once	// Avoid recurive #include issues

#include "stdafx.hpp"

/*
* This basis only normalizes the features from the environment and then passes them on. It doesn't compute additional features
*/

class IdentityBasis : public FeatureGenerator
{
public:
	////////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////////
	IdentityBasis(int inputDimension, const Eigen::VectorXd& inputLowerBound, const Eigen::VectorXd& inputUpperBound);

	////////////////////////////////////////////////////////////////
	// Functions required by FeatureGenerator parent class
	////////////////////////////////////////////////////////////////
	std::string getName() const override;
	void generateFeatures(const Eigen::VectorXd& in, Eigen::VectorXd& outBuff) override;
	int getNumOutputs() const override;

private:
	int inputDimension;					// The number of input features
	Eigen::VectorXd inputLowerBound;	// Minimum value for each input element
	Eigen::VectorXd inputUpperBound;	// Maximum value for each input element
	Eigen::ArrayXd inputRange;			// inputUpperBound - inputLowerBound. Only computed once for efficiency	
};