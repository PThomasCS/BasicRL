#pragma once	// Avoid recursive #include issues

#include "stdafx.hpp"

class AlGridworld687 : public Environment
{
public:
    ////////////////////////////////////////////////////////////////
    // Constructor
    ////////////////////////////////////////////////////////////////
    AlGridworld687();			// Default

    ////////////////////////////////////////////////////////////////
    // Functions for getting properties of the environment
    ////////////////////////////////////////////////////////////////
    int getObservationDimension() const override;				// Get the dimension of the observation vector
    int getNumActions() const override;							// Get the number of actions
    double getGamma() const override;							// Get the discount factor
    int getRecommendedEpisodeLength() const override;			// Get recommended episode length. Episodes might be terminated after this many time steps external to the environment (episodeOver will not necessarily return true).
    int getRecommendedMaxEpisodes() const override;				// Get the recommended maximum number of episodes for an agent lifetime for this environment.
    std::string getName() const override;						// Get the name of the environment
    Eigen::VectorXd getObservationLowerBound() const override;	// Get a lower bound on each observation feature
    Eigen::VectorXd getObservationUpperBound() const override;	// Get a lower bound on each observation feature

    ////////////////////////////////////////////////////////////////
    // Functions for interacting with the environment
    ////////////////////////////////////////////////////////////////
    void newEpisode(std::mt19937_64& generator) override;			// new episode
    void getObservation(std::mt19937_64& generator, Eigen::VectorXd& buff) const override;		// get observation
    bool hitObstacle(int realAction) const;                         // checks if (s, a) transition will lead to an obstacle state
    double step(int action, std::mt19937_64& generator) override;	// step from time (t) to time (t+1), where the agent selects action 'a'
    bool episodeOver(std::mt19937_64& generator) const override;	// query whether the episode is over (only call once per time step).

private:
    int size;
    int x;
    int y;
};