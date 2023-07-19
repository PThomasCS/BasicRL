#pragma once	// Avoid recursive #include issues

#include "stdafx.hpp"

class QLambda : public Agent
{
public:
    ////////////////////////////////////////////////////////////////
    // Constructor
    ////////////////////////////////////////////////////////////////
    QLambda(int observationDimension, int numActions, double alpha, double lambda, double epsilon, double gamma, FeatureGenerator* phi);

    ////////////////////////////////////////////////////////////////
    // Functions for getting properties of the agent
    ////////////////////////////////////////////////////////////////
    std::string getName() const override;											// Get the name of the environment
    bool trainBeforeAPrime() const override;										// Should train be called before or after aPrime is sampled?

    ////////////////////////////////////////////////////////////////
    // Functions for interacting with the agent, except for train
    ////////////////////////////////////////////////////////////////
    void newEpisode(std::mt19937_64& generator) override;							// new episode
    int getAction(const Eigen::VectorXd& observation, std::mt19937_64& generator) override;// get action
    void trainEpisodeEnd(const Eigen::VectorXd& observation, const int action, const double reward, std::mt19937_64& generator) override;	// Train when the next state is the terminal absorbing / is after the episode ended.

    ////////////////////////////////////////////////////////////////
    // Training functions. One of the following must be implemented
    ////////////////////////////////////////////////////////////////
    void train(const Eigen::VectorXd& observation, const int curAction, const double reward, const Eigen::VectorXd& newObservation, std::mt19937_64& generator) override;

public:
    int observationDimension;
    int numActions;
    double alpha;
    double lambda;
    double epsilon;
    double gamma;
    FeatureGenerator* phi;

    Eigen::MatrixXd w;	// Weights for the q-approximation
    Eigen::MatrixXd e;	// Eligibility traces

    Eigen::VectorXd curFeatures;
    Eigen::VectorXd newFeatures;
    bool curFeaturesInit;
};