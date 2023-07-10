#pragma once	// Avoid recursive #include issues

#include "stdafx.hpp"

class AlActorCritic : public Agent
{
public:
    ////////////////////////////////////////////////////////////////
    // Constructor
    ////////////////////////////////////////////////////////////////
    AlActorCritic(int observationDimension, int numActions, double alpha, double lambda, double epsilon, double gamma, double sigma, FeatureGenerator* phi);

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
    double beta;
    double lambda;
    double gamma;
    double sigma;
    FeatureGenerator* phi;

    Eigen::VectorXd w;	    // Weights for the v-approximation
    Eigen::MatrixXd eV;	    // Eligibility traces

    Eigen::MatrixXd theta;	// Weights for the policy
    Eigen::MatrixXd eTheta;	// Eligibility traces
    Eigen::VectorXd actor;  // Policy

    Eigen::VectorXd curFeatures;
    Eigen::VectorXd newFeatures;
    bool curFeaturesInit;
};