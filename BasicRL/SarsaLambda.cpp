#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

SarsaLambda::SarsaLambda(int observationDimension, int numActions, double alpha, double lambda, double epsilon, double gamma, FeatureGenerator * phi)
{
	// Copy over arguments
	this->observationDimension = observationDimension;
	this->numActions = numActions;
	this->alpha = alpha;
	this->lambda = lambda;
	this->epsilon = epsilon;
	this->gamma = gamma;
	this->phi = phi;

	// Indicate that we have not loaded curFeatures
	curFeaturesInit = false;

	// Initialize the weights and e-traces
	e = w = MatrixXd::Zero(numActions, phi->getNumOutputs());
}

string SarsaLambda::getName() const
{
	return "Linear Sarsa(Lambda) with lambda = " + to_string(lambda);
}

bool SarsaLambda::trainBeforeAPrime() const
{
	return false;
}

void SarsaLambda::newEpisode(std::mt19937_64& generator)
{
	curFeaturesInit = false;	// We have not loaded curFeatures when the next train call happens.
	e.setZero();				// Clear the eligibility traces
}

int SarsaLambda::getAction(const Eigen::VectorXd& observation, std::mt19937_64& generator)
{
	// Load curFeatures and newFeatures, even if we are going to explore and not use them. They may be used in training.
	VectorXd qValues(numActions);
	if (!curFeaturesInit)	// If we have not initialized curFeatures, use them
	{
		phi->generateFeatures(observation, curFeatures);
		qValues = w * curFeatures;
		curFeaturesInit = true;	// We have now loaded curFeatures
	}
	else
	{
		phi->generateFeatures(observation, newFeatures);
		qValues = w * newFeatures;
	}

	// Handle epsilon greedy exploration
	bernoulli_distribution explorationDistribution(epsilon);
	bool explore = explorationDistribution(generator);
	if (explore)
	{
		uniform_int_distribution<int> uniformActionDistribution(0, numActions - 1);
		int result = uniformActionDistribution(generator);
		return result;
	}

	// If we get here, we aren't exploring! Return an action that achieves the maximum q-value
	return maxIndex(qValues, generator);
}

void SarsaLambda::trainEpisodeEnd(const Eigen::VectorXd& observation, const int action, const double reward, std::mt19937_64& generator)
{
	// Compute the TD-error
	double delta = reward - w.row(action).dot(curFeatures);	// We already computed the features for "observation" at a getAction call and stored them in curFeatures

	// Update the e-traces
	e = gamma * lambda * e;
	e.row(action) += curFeatures;

	// Update the weights
	w = w + alpha * delta * e;
}

void SarsaLambda::train(const Eigen::VectorXd& observation, const int curAction, const double reward, const Eigen::VectorXd& newObservation, const int newAction, std::mt19937_64& generator)
{
	// Compute the TD-error
	double delta = reward + gamma * w.row(newAction).dot(newFeatures) - w.row(curAction).dot(curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures

	// Update the e-traces
	e = gamma * lambda * e;
	e.row(curAction) += curFeatures;

	// Update the weights
	w = w + alpha * delta * e;

	// Move newFeatures into curFeatures
	curFeatures = newFeatures;
}