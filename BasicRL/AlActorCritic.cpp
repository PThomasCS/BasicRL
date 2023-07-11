#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

AlActorCritic::AlActorCritic(int observationDimension, int numActions, double alpha, double beta, double lambda, double gamma, double sigma, FeatureGenerator * phi)
{
    // Copy over arguments
    this->observationDimension = observationDimension;
    this->numActions = numActions;
    this->alpha = alpha;
    this->beta = beta;
    this->lambda = lambda;
    this->gamma = gamma;
    this->sigma = sigma; // For softmax (sigma -> 0, action selection is more stochastic; sigma -> inf, action selection is more deterministic)
    this->phi = phi;

    // Indicate that we have not loaded curFeatures
    curFeaturesInit = false;

    // Initialize the e-traces and weights for the critic and actor
    eTheta = theta = MatrixXd::Zero(numActions, phi->getNumOutputs());
    eV = w = VectorXd::Zero(phi->getNumOutputs());

    // Initialize the policy vector
    actor = VectorXd::Zero(numActions);
}

string AlActorCritic::getName() const
{
    return "Actor-Critic with lambda = " + to_string(lambda);
}

bool AlActorCritic::trainBeforeAPrime() const
{
    return true;
}

void AlActorCritic::newEpisode(std::mt19937_64& generator)
{
    curFeaturesInit = false;	// We have not loaded curFeatures when the next train call happens.
    eV.setZero();				// Clear the eligibility traces for the critic
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

    for (int i = 0; i < numActions; i++)
    { // exp(theta(s, a) = exp(dot product of theta.row(s) and phi(s)) => real number
        double expThetaSA = exp(sigma * theta.row(i).dot(curFeatures));
        // Assign computed exp(theta(s, a) to actor at idx a
        actor(i) = expThetaSA;
    }

    // Normalize
    actor /= actor.sum();

    discrete_distribution<> distPSAs(actor.data(), actor.data() + actor.size());

    return distPSAs(generator);
}

void AlActorCritic::trainEpisodeEnd(const Eigen::VectorXd& observation, const int action, const double reward, std::mt19937_64& generator)
{
    // Critic update
    // Update the e-traces for the critic
    eV = gamma * lambda * eV + curFeatures;

    // Compute the TD-error
    double delta = reward - w.dot(curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures

    // Update the weights for the critic
    w = w + alpha * delta * eV;

    // Actor update
    // Update the e-traces for the actor

    for (int i = 0; i < numActions; i++)
    {
        if (i == action)
        {
            eTheta.row(i) = gamma * lambda * eTheta.row(i).transpose() + ((1.0 - actor(i)) * curFeatures);
        }
        else
        {
            eTheta.row(i) = gamma * lambda * eTheta.row(i).transpose()  + (-1.0 * actor(i)) * curFeatures;
        }
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

    eV = gamma * lambda * eV + curFeatures;

    // Compute the TD-error
    double delta = reward + gamma * w.dot(newFeatures) - w.dot(curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures


    // Update the weights for the critic
    w = w + alpha * delta * eV;

    // Actor update
    // Update the e-traces for the actor

    for (int i = 0; i < numActions; i++)
    {
        if (i == curAction)
        {
            eTheta.row(i) = gamma * lambda * eTheta.row(i).transpose() + ((1.0 - actor(i)) * curFeatures);
        }
        else
        {
            eTheta.row(i) = gamma * lambda * eTheta.row(i).transpose()  + (-1.0 * actor(i)) * curFeatures;
        }
    }

    // Update the weights for the actor
    theta = theta + beta * delta * eTheta;

    // Move newFeatures into curFeatures
    curFeatures = newFeatures;
}
