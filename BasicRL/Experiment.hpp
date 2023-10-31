//#pragma once	// Avoid recurive #include issues
//
//#include "stdafx.hpp"
//
//class Experiment
//{
//public:
//	////////////////////////////////////////////////////////////////
//	// Constructor
//	////////////////////////////////////////////////////////////////
//	Experiment();	
//
//	////////////////////////////////////////////////////////////////
//	// Functions tp add parameters to experiment
//	////////////////////////////////////////////////////////////////
//	std::string addAgentName();
//
//	int getObservationDimension() const override;				// Get the dimension of the observation vector
//	int getNumActions() const override;							// Get the number of actions
//	double getGamma() const override;							// Get the discount factor
//	int getRecommendedEpisodeLength() const override;			// Get recommended episode length. Episodes might be terminated after this many time steps external to the environment (episodeOver will not necessarily return true).
//	int getRecommendedMaxEpisodes() const override;				// Get the recommended maximum number of episodes for an agent lifetime for this environment.
//	std::string getName() const override;						// Get the name of the environment
//	Eigen::VectorXd getObservationLowerBound() const override;	// Get a lower bound on each observation feature
//	Eigen::VectorXd getObservationUpperBound() const override;	// Get a lower bound on each observation feature
//
//	////////////////////////////////////////////////////////////////
//	// Functions for interacting with the environment
//	////////////////////////////////////////////////////////////////
//	void newEpisode(std::mt19937_64& generator) override;			// new episode
//	void getObservation(std::mt19937_64& generator, Eigen::VectorXd& buff) const override;		// get observation
//	double step(int action, std::mt19937_64& generator) override;	// step from time (t) to time (t+1), where the agent selects action 'a'
//	bool episodeOver(std::mt19937_64& generator) const override;	// query whether the episode is over (only call once per time step).
//
//private:
//	std::vector<int> numHyperParamExperiments;                       // Length = total number of sets of hyperparameters
//	std::vector<int> numTrialsInExperiment;							// Length = total number of agent-environment pairs.   numTrialsInExperiment[i] is the number of trials to run for the i'th agent-environment pair.
//	std::vector<int> experimentIDs;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
//	std::vector<int> trialCounts;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
//	std::vector<std::string> agentNames;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
//	std::vector<std::string> envNames;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
//	std::vector<std::string> featureGenNames;								// Length = total number of trials (sum of elements in numTrialsInExperiment)																										
//	std::vector<std::unordered_map<std::string, int>> featureGenParameters;	// Length = total number of trials (sum of elements in numTrialsInExperiment). i'th element is a map of parameters for this agent-environment pair's feature generator
//	std::vector<std::unordered_map<std::string, double>> hyperParameters;		// Length = total number of trials (sum of elements in numTrialsInExperiment). i'th element is a map of hyperparameters for this agent-environment pair
//};