#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

Reinforce::Reinforce(int observationDimension, int numActions, double alpha, double gamma, FeatureGenerator* phi, bool includeExtraGamma)
{
    // Copy over arguments
    this->observationDimension = observationDimension;
    this->numActions = numActions;
    this->alpha = alpha;
    this->gamma = gamma;
    this->phi = phi;
    this->includeExtraGamma = includeExtraGamma;

    // Allocate memory for curFeatures and newFeatures
    int numFeatures = phi->getNumOutputs();
    curFeatures = new VectorXd(numFeatures);
    newFeatures = new VectorXd(numFeatures);

    // Indicate that we have not loaded curFeatures
    curFeaturesInit = false;

    // Initialize the e-traces and weights for the critic and actor
    psi = theta = grad = MatrixXd::Zero(numActions, numFeatures);

    // Initialize the policy vector
    actionProbabilities.resize(numActions);
}

// Called automatically when the object is destroyed - do not manually call this function!
Reinforce::~Reinforce()
{
    // Clean up the memory that we allocated
    delete curFeatures;
    delete newFeatures;
}

string Reinforce::getName() const
{
    return "REINFORCE with alpha = " + to_string(alpha);
}

bool Reinforce::trainBeforeAPrime() const
{
    return true;
}

void Reinforce::newEpisode(std::mt19937_64& generator)
{
    curFeaturesInit = false;	// We have not loaded curFeatures when the next train call happens.
    rewards.clear();
    returns.clear();
    psis.clear();
    psi.setZero();
    grad.setZero();
}

int Reinforce::getAction(const Eigen::VectorXd& observation, std::mt19937_64& generator)
{
    // Handle softmax action selection
    if (!curFeaturesInit)	// If we have not initialized curFeatures, intialize them
    {
        phi->generateFeatures(observation, *curFeatures);
        curFeaturesInit = true;
    }

    //actionProbabilities = (theta * (*curFeatures)).array().exp();

    //// Normalize
    //actionProbabilities /= actionProbabilities.sum();

    //// Our own just to check speed!
    //return randp(actionProbabilities, generator);

    // Code from Chat GPT (log trick)
        // Calculate unnormalized log probabilities
    Eigen::VectorXd logProbabilities = theta * (*curFeatures);

    // Calculate the maximum value among the log probabilities
    double maxLogProbability = logProbabilities.maxCoeff();

    // Subtract the maximum value from all log probabilities to avoid overflow
    logProbabilities.array() -= maxLogProbability;

    // Calculate the log of the sum of exponentials of the adjusted log probabilities
    double logSumExp = logProbabilities.array().exp().sum();

    // Calculate the normalized probabilities using the log-sum-exp trick
    actionProbabilities = (logProbabilities.array() - logSumExp).exp();

    // Normalize the probabilities
    actionProbabilities /= actionProbabilities.sum();

    // Our own just to check speed!
    return randp(actionProbabilities, generator);
}

void Reinforce::trainEpisodeEnd(const Eigen::VectorXd& observation, const int action,
    const double reward, std::mt19937_64& generator)
{
    // Compute the discounted returns from each time step in the episode
    int episodeLength = (int)rewards.size();
    returns.resize(episodeLength);
    
    // Gt will be used to store the return from time t
    double Gt = 0.0;

    // Computing Gt and also the portion of the policy gradient estimate from time t
    double gammaTerm = 1.0;
    for (int t = episodeLength - 1; t >= 0; t--)
    {
        Gt = rewards[t] + gamma * Gt;
        if (includeExtraGamma)
            gammaTerm = pow(gamma, t);      // This is inefficient, but this variant of REINFORCE shouldn't usually be used. If when profiling this is a real chokepoint, consider pre-computing the powers of gamma in a new vector with a loop before this one.
        grad += gammaTerm * Gt * psis[t];
    }

    // Update the policy weights
    theta = theta + alpha * grad;
}

void Reinforce::train(const Eigen::VectorXd& observation, const int curAction, const double reward, const Eigen::VectorXd& newObservation, std::mt19937_64& generator)
{   
    // Save the reward at time step t to r
    rewards.push_back(reward);

    // Calculate the derivatives w.r.t theta and save them to an array at time step t
    for (int i = 0; i < numActions; i++)
        psi.row(i) = (((i == curAction ? 1.0 : 0.0) - actionProbabilities(i)) * (*curFeatures));

    // Store the compatible features that we have computed.
    psis.push_back(psi);
         
    // Generate NnewFeatures for newObservation
    phi->generateFeatures(newObservation, *newFeatures);

    // Move newFeatures into curFeatures
    swap(curFeatures, newFeatures);
}