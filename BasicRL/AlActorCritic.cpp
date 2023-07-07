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

    for (int i = 0; i < numActions; i++)
    {
        double thetaSA = theta.row(i).dot(observation);
        pSAs(i) = thetaSA;
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
    double delta = reward + gamma * w.dot(newFeatures) - w.dot(curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures

    // Update the weights for the critic
    w = w + alpha * delta * eV;

    // Actor update
    // Update the e-traces for the actor

    eTheta = gamma * lambda * eTheta;
    eTheta.row(action) += curFeatures;

    // Update the weights for the actor
    for (int i = 0; i < theta.size(); i++)
    {
        if (i == action)
            theta.row(i) = theta.row(i) + alpha * delta * eTheta.row(i) + (1.0 - pSAs(i)) * curFeatures;
        else
        {
            theta.row(i) = theta.row(i) + alpha * delta * eTheta.row(i) + (-1.0 * pSAs(i)) * curFeatures;
        }
    }
}

void AlActorCritic::train(const Eigen::VectorXd& observation, const int curAction, const double reward, const Eigen::VectorXd& newObservation, std::mt19937_64& generator)
{
    // Critic update
    // Update the e-traces for the critic
    eV = gamma * lambda * eV + curFeatures;

    // Compute the TD-error
    double delta = reward + gamma * w.dot(newFeatures) - w.dot(curFeatures);	// We already computed the features for "curObservation" at a getAction call and stored them in curFeatures, and newObservation-->newFeatures

    // Update the weights for the critic
    w = w + alpha * delta * eV;

    // Actor update
    // Update the e-traces for the actor

    eTheta = gamma * lambda * eTheta;
    eTheta.row(curAction) += curFeatures;

    // Update the weights for the actor
    for (int i = 0; i < theta.size(); i++)
    {
        if (i == curAction)
            theta.row(i) = theta.row(i) + alpha * delta * eTheta.row(i) + (1.0 - pSAs(i)) * curFeatures;
        else
        {
            theta.row(i) = theta.row(i) + alpha * delta * eTheta.row(i) + (-1.0 * pSAs(i)) * curFeatures;
        }
    }

    // Move newFeatures into curFeatures
    curFeatures = newFeatures;
}
