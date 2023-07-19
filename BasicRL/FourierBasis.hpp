#pragma once	// Avoid recurive #include issues

#include "stdafx.hpp"

class FourierBasis : public FeatureGenerator
{
public:
	////////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////////
	FourierBasis(int inputDimension, const Eigen::VectorXd& inputLowerBound, const Eigen::VectorXd& inputUpperBound, int iOrder, int dOrder);

	////////////////////////////////////////////////////////////////
	// Functions required by FeatureGenerator parent class
	////////////////////////////////////////////////////////////////
	void generateFeatures(const Eigen::VectorXd& in, Eigen::VectorXd& outBuff) override;
	int getNumOutputs() const override;

private:
	int iOrder;							// "independent" order
	int dOrder;							// "dependent" order
	int numOutputs;						// Total number of outputs
	int inputDimension;					// The number of input features
	Eigen::MatrixXd c;					// Coefficient matrix (called 'c' in the Fourier Basis paper)
	Eigen::VectorXd inputLowerBound;	// Minimum value for each input element
	Eigen::VectorXd inputUpperBound;	// Maximum value for each input element
	Eigen::ArrayXd inputRange;			// inputUpperBound - inputLowerBound. Only computed once for efficiency

	// This variable is used in generateFeatures, which is called frequently. Don't create/allocate it every time.
	Eigen::VectorXd normalizedInput;
};