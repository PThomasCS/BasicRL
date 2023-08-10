#pragma once	// Avoid recurive #include issues

#include "stdafx.hpp"

class FeatureGenerator
{
public:
	virtual std::string getName() const = 0;
	virtual void generateFeatures(const Eigen::VectorXd& in, Eigen::VectorXd& outBuff) = 0;	// Generate features for the provided input. Store the result in outBuff
	virtual int getNumOutputs() const = 0;										// Get the number of outputs this feature generator produces
};