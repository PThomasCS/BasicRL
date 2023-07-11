#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

AlActorCritic::AlActorCritic(int observationDimension, int numActions, double alpha, double beta,
    double lambda, double gamma, FeatureGenerator * phi)
{
    // Copy over arguments
    this->observationDimension = observationDimension;
    this->numActions = numActions;
    this->alpha = alpha;
    this->beta = beta;
    this->lambda = lambda;
    this->gamma = gamma;
    this->phi = phi;

    // Indicate that we have not loaded curFeatures
    curFeaturesInit = false;

    // Initialize the e-traces and weights for the critic and actor
    eTheta = theta = MatrixXd::Zero(numActions, phi->getNumOutputs());
    ew = w = VectorXd::Zero(phi->getNumOutputs());

    // Initialize the policy vector
    actionProbabilities.resize(numActions);
}

string AlActorCritic::getName() const
{
    return "Alexandra Actor-Critic with lambda = " + to_string(lambda);
}

bool AlActorCritic::trainBeforeAPrime() const
{
    return true;
}

void AlActorCritic::newEpisode(std::mt19937_64& generator)
{
    curFeaturesInit = false;	// We have not loaded curFeatures when the next train call happens.
    ew.setZero();				// Clear the eligibility traces for the critic
    eTheta.setZero();		    // Clear the eligibility traces for the actor
}

int AlActorCritic::getAction(const Eigen::VectorXd& observation, std::mt19937_64& generator)
{
    // Handle softmax action selection
    if (!curFeaturesInit)	// If we have not initialized curFeatures, use them
    {
        phi->generateFeatures(observation, curFeatures);
        curFeaturesInit = true;
    }

    actionProbabilities = (theta*curFeatures).array().exp();

    // Normalize
    actionProbabilities /= actionProbabilities.sum();

    discrete_distribution<int> distPSAs(actionProbabilities.data(), actionProbabilities.data() + actionProbabilities.size());
    return distPSAs(generator);
}

void AlActorCritic::trainEpisodeEnd(const Eigen::VectorXd& observation, const int action, const double reward, std::mt19937_64& generator)
{
    // Critic update
    // Update the e-traces for the critic
    ew = gamma * lambda * ew + curFeatures;

    // Compute the TD-error
    double delta = reward - w.dot(curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures

    // Update the weights for the critic
    w = w + alpha * delta * ew;

    // Actor update
    // Update the e-traces for the actor
    eTheta = gamma * lambda * eTheta;
    for (int i = 0; i < numActions; i++)
    {
        eTheta.row(i) += (((i == action ? 1.0 : 0.0) - actionProbabilities(i)) * curFeatures);
        /*
        if (i == action)
            eTheta.row(i) += ((1.0 - actionProbabilities(i)) * curFeatures);
        else
            eTheta.row(i) += (-actionProbabilities(i)) * curFeatures;
        */
    }

    // Update the weights for the actor
    // Add beta?
    theta = theta + beta * delta * eTheta;
}

void AlActorCritic::train(const Eigen::VectorXd& observation, const int curAction, const double reward, const Eigen::VectorXd& newObservation, std::mt19937_64& generator)
{
    phi->generateFeatures(newObservation, newFeatures);

    // Critic update
    // Update the e-traces for the critic

    ew = gamma * lambda * ew + curFeatures;

    // Compute the TD-error
    double delta = reward + gamma * w.dot(newFeatures) - w.dot(curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures


    // Update the weights for the critic
    w = w + alpha * delta * ew;

    // Actor update
    // Update the e-traces for the actor
    eTheta = gamma * lambda * eTheta;
    for (int i = 0; i < numActions; i++)
        eTheta.row(i) +=  (((i == curAction ? 1.0 : 0.0) - actionProbabilities(i)) * curFeatures);
    
    // Update the weights for the actor
    theta = theta + beta * delta * eTheta;

    // Move newFeatures into curFeatures
    curFeatures = newFeatures;
}
