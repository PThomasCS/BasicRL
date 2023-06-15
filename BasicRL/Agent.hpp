#pragma once	// Avoid recurive #include issues

#include "stdafx.hpp"

/*
* Agent Class
*
* This class describes the interface of environments used for online reinforcement learning.
* It assumes that the state is a VectorXd and that the actions are integers.
*/
class Agent
{
public:
	////////////////////////////////////////////////////////////////
	// Functions for getting properties of the agent
	////////////////////////////////////////////////////////////////
	virtual std::string getName() const = 0;						// Get the name of the environment
	virtual bool trainBeforeAPrime() const = 0;						// Should train be called before or after aPrime is sampled?

	////////////////////////////////////////////////////////////////
	// Functions for interacting with the agent, except for train
	////////////////////////////////////////////////////////////////
	virtual void newEpisode(std::mt19937_64& generator) = 0;								// new episode
	virtual int getAction(const Eigen::VectorXd& observation, std::mt19937_64& generator) = 0;		// get action
	virtual void trainEpisodeEnd(const Eigen::VectorXd& observation, const int action, const double reward, std::mt19937_64& generator) = 0;	// Train when the next state is the terminal absorbing / is after the episode ended.

	////////////////////////////////////////////////////////////////
	// Training functions. One of the following must be implemented
	////////////////////////////////////////////////////////////////
	virtual void train(
		const Eigen::VectorXd& observation, 
		const int action, 
		const double reward, 
		const Eigen::VectorXd& newObservation,
		const int newAction, 
		std::mt19937_64& generator)
	{
		errorExit("Error: train(o,a,r,oPrime,aPrime) called for an agent without this function. Agent type = " + getName() + ".");
	}
	
	virtual void train(
		const Eigen::VectorXd& observation,
		const int action, 
		const double reward, 
		const Eigen::VectorXd& newObservation,
		std::mt19937_64& generator)
	{
		errorExit("Error: train(o,a,r,oPrime) called for an agent without this function. Agent type = " + getName() + ".");
	}
};