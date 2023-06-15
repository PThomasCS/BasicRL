#pragma once	// Avoid recurive #include issues

#include "stdafx.hpp"

/*
* Environment Class
*
* This class describes the interface of environments used for online reinforcement learning.
* It assumes that the state is a VectorXd and that the actions are integers.
* This is a "pure abstract class" in that you do not create Environment objects - you create
* Gridworld, MountainCar, etc. objects, which follow this specification.
*/
class Environment
{
public:
	////////////////////////////////////////////////////////////////
	// Functions for getting properties of the environment
	////////////////////////////////////////////////////////////////
	virtual int getObservationDimension() const = 0;				// Get the dimension of the observation vector
	virtual int getNumActions() const = 0;							// Get the number of actions
	virtual double getGamma() const = 0;							// Get the discount factor
	virtual int getRecommendedEpisodeLength() const = 0;			// Get recommended episode length. Episodes might be terminated after this many time steps external to the environment (episodeOver will not necessarily return true).
	virtual int getRecommendedMaxEpisodes() const = 0;				// Get the recommended maximum number of episodes for an agent lifetime for this environment.
	virtual std::string getName() const = 0;						// Get the name of the environment
	virtual Eigen::VectorXd getObservationLowerBound() const = 0;	// Get a lower bound on each observation feature
	virtual Eigen::VectorXd getObservationUpperBound() const = 0;	// Get a lower bound on each observation feature
	
	////////////////////////////////////////////////////////////////
	// Functions for interacting with the environment
	////////////////////////////////////////////////////////////////
	virtual void newEpisode(std::mt19937_64& generator) = 0;			// new episode
	virtual void getObservation(std::mt19937_64& generator, Eigen::VectorXd & buff) const = 0;	// get observation
	virtual double step(int action, std::mt19937_64& generator) = 0;	// step from time (t) to time (t+1), where the agent selects action 'a', return the resulting reward
	virtual bool episodeOver(std::mt19937_64& generator) const = 0;		// query whether the episode is over (only call once per time step).
};