#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

ExpectedSarsaLambda::ExpectedSarsaLambda(int observationDimension, int numActions, double alpha, double lambda,
    double epsilon, double gamma, FeatureGenerator* phi)
{
    // Copy over arguments
    this->observationDimension = observationDimension;
    this->numActions = numActions;
    this->alpha = alpha;
    this->lambda = lambda;
    this->epsilon = epsilon;
    this->gamma = gamma;
    this->phi = phi;

    // Allocate memory for curFeatures and newFeatures
    int numFeatures = phi->getNumOutputs();
    curFeatures = new VectorXd(numFeatures);
    newFeatures = new VectorXd(numFeatures);

    // Indicate that we have not loaded curFeatures
    curFeaturesInit = false;

    // Initialize the weights and e-traces
    e = w = MatrixXd::Zero(numActions, numFeatures);
}

// Called automatically when the object is destroyed - do not manually call this function!
ExpectedSarsaLambda::~ExpectedSarsaLambda()
{
    // Clean up the memory that we allocated
    delete curFeatures;
    delete newFeatures;
}

string ExpectedSarsaLambda::getName() const
{
    return "Expected Sarsa (Lambda) with lambda = " + to_string(lambda) + ", alpha = " + to_string(alpha) + ", epsilon = " + to_string(epsilon);
}

bool ExpectedSarsaLambda::trainBeforeAPrime() const
{
    return true;
}

void ExpectedSarsaLambda::newEpisode(std::mt19937_64& generator)
{
    curFeaturesInit = false;	// We have not loaded curFeatures when the next train call happens.
    e.setZero();				// Clear the eligibility traces
}

int ExpectedSarsaLambda::getAction(const Eigen::VectorXd& observation, std::mt19937_64& generator)
{
    // Load curFeatures, even if we are going to explore and not use them. They may be used in training.
    if (!curFeaturesInit)	// If we have not initialized curFeatures, use them
    {
        phi->generateFeatures(observation, *curFeatures); // *curFeatures means, pass the object that curFeatures points to.
        curFeaturesInit = true;
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

    VectorXd qValues(numActions);
    qValues = w * (*curFeatures);

    return maxIndex(qValues, generator);
}

void ExpectedSarsaLambda::trainEpisodeEnd(const Eigen::VectorXd& observation, const int action,
    const double reward, std::mt19937_64& generator)
{
    // Compute the TD-error
    double delta = reward - w.row(action).dot(*curFeatures);	// We already computed the features for "observation" at a getAction call and stored them in curFeatures

    // Update the e-traces
    e = gamma * lambda * e;
    e.row(action) += *curFeatures;

    // Update the weights
    w = w + alpha * delta * e;
}

void ExpectedSarsaLambda::train(const Eigen::VectorXd& observation, const int curAction, const double reward, const Eigen::VectorXd& newObservation, std::mt19937_64& generator)
{
    phi->generateFeatures(newObservation, *newFeatures);

    // Calculate the expectation of QValues of sPrime, aPrime
    // Calculate the QValues and find the indices of best actions
    VectorXd newQValues(numActions);
    newQValues = w * (*newFeatures);

    vector<int> bestActions = maxIndices(newQValues);
    int numBestActions = (int)bestActions.size();
    
    // Calculate action probabilities
    // If we explore (with probability epsilon), then all actions are sampled uniformly
    VectorXd actionProbabilities(numActions);
    actionProbabilities.setConstant(epsilon * (1.0 / numActions));

    // Add to all best actions uniform probability of sampling that best action if we are not exploring (with probability 1 - epsilon)
    for (int i = 0; i < numBestActions; i++)
    {
        int bestActionIdx = bestActions[i];
        actionProbabilities[bestActionIdx] += (1.0 - epsilon) * (1.0 / numBestActions);
    }

    // Compute the TD-error
    // The expectation of QValues of sPrime, aPrime is calculated by qValues.dot(actionProbabilities)
    double delta = reward + gamma * newQValues.dot(actionProbabilities) - w.row(curAction).dot(*curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures

    // Update the e-traces
    e = gamma * lambda * e;
    e.row(curAction) += *curFeatures;

    // Update the weights
    w = w + alpha * delta * e;

    // Move newFeatures into curFeatures
    swap(curFeatures, newFeatures);
}