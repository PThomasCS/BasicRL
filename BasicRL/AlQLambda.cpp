#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

AlQLambda::AlQLambda(int observationDimension, int numActions, double alpha, double lambda, double epsilon, double gamma, FeatureGenerator * phi)
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

string AlQLambda::getName() const
{
    return "Q(Lambda) with lambda = " + to_string(lambda);
}

bool AlQLambda::trainBeforeAPrime() const
{
    return true;
}

void AlQLambda::newEpisode(std::mt19937_64& generator)
{
    curFeaturesInit = false;	// We have not loaded curFeatures when the next train call happens.
    e.setZero();				// Clear the eligibility traces
}

int AlQLambda::getAction(const Eigen::VectorXd& observation, std::mt19937_64& generator)
{
    // Handle epsilon greedy exploration
    bernoulli_distribution explorationDistribution(epsilon);
    bool explore = explorationDistribution(generator);
    if (explore)
    {
        uniform_int_distribution<int> uniformActionDistribution(0, numActions - 1);
        int result = uniformActionDistribution(generator);
        return result;
    }

    // If we get here, we're not exploring. Get the q-values
    VectorXd qValues(numActions); // Deleted features var

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

    // Return an action that achieves the maximum q-value
    return maxIndex(qValues, generator);
}

void AlQLambda::trainEpisodeEnd(const Eigen::VectorXd& observation, const int action, const double reward, std::mt19937_64& generator)
{
    // Compute the TD-error
    double delta = reward - w.row(action).dot(curFeatures);	// We already computed the features for "observation" at a getAction call and stored them in curFeatures

    // Update the e-traces
    e = gamma * lambda * e;
    e.row(action) += curFeatures;

    // Update the weights
    w = w + alpha * delta * e;
}

void AlQLambda::train(const Eigen::VectorXd& observation, const int curAction, const double reward, const Eigen::VectorXd& newObservation, std::mt19937_64& generator)
{

    // Compute the TD-error
    double delta = reward + gamma * (w * newFeatures).maxCoeff() - w.row(curAction).dot(curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures

    // Update the e-traces
    e = gamma * lambda * e;
    e.row(curAction) += curFeatures;

    // Update the weights
    w = w + alpha * delta * e;

    // Move newFeatures into curFeatures
    curFeatures = newFeatures;
}