#pragma once	// Avoid recursive #include issues

#include "stdafx.hpp"

class AlMountainCar : public Environment
{
public:
    ////////////////////////////////////////////////////////////////
    // Constructor
    ////////////////////////////////////////////////////////////////
    AlMountainCar();

    ////////////////////////////////////////////////////////////////
    // Functions for getting properties of the environment
    ////////////////////////////////////////////////////////////////
    virtual int getObservationDimension() const override;				// Get the dimension of the observation vector
    virtual int getNumActions() const override;							// Get the number of actions
    virtual double getGamma() const override;							// Get the discount factor
    virtual int getRecommendedEpisodeLength() const override;			// Get recommended episode length. Episodes might be terminated after this many time steps external to the environment (episodeOver will not necessarily return true).
    virtual int getRecommendedMaxEpisodes() const override;				// Get the recommended maximum number of episodes for an agent lifetime for this environment.
    virtual std::string getName() const override;						// Get the name of the environment
    virtual Eigen::VectorXd getObservationLowerBound() const override;	// Get a lower bound on each observation feature
    virtual Eigen::VectorXd getObservationUpperBound() const override;	// Get a lower bound on each observation feature

    ////////////////////////////////////////////////////////////////
    // Functions for interacting with the environment
    ////////////////////////////////////////////////////////////////
    virtual void newEpisode(std::mt19937_64& generator) override;			                            // new episode
    virtual void getObservation(std::mt19937_64& generator, Eigen::VectorXd & buff) const override;	    // get observation
    virtual double step(int action, std::mt19937_64& generator) override;	                            // step from time (t) to time (t+1), where the agent selects action 'a', return the resulting reward
    virtual bool episodeOver(std::mt19937_64& generator) const override;		                        // query whether the episode is over (only call once per time step).

private:
    double x;
    double v;
};