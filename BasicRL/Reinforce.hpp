#pragma once	// Avoid recursive #include issues

#include "stdafx.hpp"

class Reinforce : public Agent
{
public:
    ////////////////////////////////////////////////////////////////
    // Constructor
    ////////////////////////////////////////////////////////////////
    Reinforce(int observationDimension, int numActions, double alpha, double gamma, FeatureGenerator* phi, bool includeExtraGamma = false);
    ~Reinforce();

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
    double epsilon;
    double gamma;
    FeatureGenerator* phi;
    bool includeExtraGamma;

    Eigen::VectorXd actionProbabilities;    // Policy

    std::vector<double> rewards;            // Rewards in the episode
    std::vector<double> returns;            // Returns from time t

    Eigen::MatrixXd theta;                  // Weights for the policy
    
    Eigen::MatrixXd psi;                    // $\partial \ln( \pi(s,a,\theta) ) / \partial \theta$ - the compatible features
    std::vector<Eigen::MatrixXd> psis;      // Vector of all psi (compatible features) for one episode. psis[t] is psi from time step t.

    Eigen::MatrixXd grad;                   // The unbiased estimate of the policy gradient from one episode

    Eigen::VectorXd* curFeatures;
    Eigen::VectorXd* newFeatures;
    bool curFeaturesInit;
};