#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

AlActorCritic::AlActorCritic(int observationDimension, int numActions, double alpha, double lambda, double epsilon, double gamma, FeatureGenerator * phi)
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

    // Initialize the e-traces and weights for the critic and actor
    eTheta = theta = MatrixXd::Zero(numActions, phi->getNumOutputs());
    eV = w = VectorXd::Zero(phi->getNumOutputs());

    // Intitialize the policy vector
    pSAs = VectorXd::Zero(numActions);
}

string AlActorCritic::getName() const
{
    return "Actor-Critic with lambda = " + to_string(lambda);
}

bool AlActorCritic::trainBeforeAPrime() const
{
    return true; // SHOULD BE TRUE!
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
    // Add sigma?
    // Fix double compute of new features

    if (!curFeaturesInit)	// If we have not initialized curFeatures, use them
    {
        phi->generateFeatures(observation, curFeatures);
        for (int i = 0; i < numActions; i++) {
            // exp(theta(s, a) = exp(dot product of theta.row(s) and phi(s)) => real number
            double expThetaSA = exp(theta.row(i).dot(curFeatures));
            // Assign computed exp(theta(s, a) to pSAs at idx a
            pSAs(i) = expThetaSA;
            curFeaturesInit = true;
        }
    }

    else
    {
        phi->generateFeatures(observation, newFeatures);
        for (int i = 0; i < numActions; i++) {
            // exp(theta(s, a) = exp(dot product of theta.row(s) and phi(s)) => real number
            double expThetaSA = exp(theta.row(i).dot(curFeatures));
            // Assign computed exp(theta(s, a) to pSAs at idx a
            pSAs(i) = expThetaSA;
        }
    }

    // Normalize
    pSAs /= pSAs.sum();

    discrete_distribution<> distPSAs(pSAs.data(), pSAs.data() + pSAs.size());

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
        if (i == action) {
            eTheta.row(i) = gamma * lambda * eTheta.row(i).transpose() + ((1.0 - pSAs(i)) * curFeatures);
        }
        else
        {
            eTheta.row(i) = gamma * lambda * eTheta.row(i).transpose()  + (-1.0 * pSAs(i)) * curFeatures;
        }
    }

    // Update the weights for the actor
    // Add beta?
    theta = theta + alpha * delta * eTheta;
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
        if (i == curAction) {
            eTheta.row(i) = gamma * lambda * eTheta.row(i).transpose() + ((1.0 - pSAs(i)) * curFeatures);
        }
        else
        {
            eTheta.row(i) = gamma * lambda * eTheta.row(i).transpose()  + (-1.0 * pSAs(i)) * curFeatures;
        }
    }

    // Update the weights for the actor
    // Add beta?
    theta = theta + alpha * delta * eTheta;

    // Move newFeatures into curFeatures
    curFeatures = newFeatures;
}
