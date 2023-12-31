/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
// START V(-3)
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.hpp"
#include <unordered_map>

using namespace std;
using namespace Eigen;

/*
* Run numTrials agent lifetimes on the provided environment. The entry in position (i,j) of the resulting matrix is the return from the j'th episode in the i'th trial.
*/

vector<MatrixXd> run(vector<Agent*> agents, vector<Environment*> environments, VectorXi maxEpisodes, VectorXi maxEpisodeLength, mt19937_64& generator, vector<int> numExperimentTrials, int numTrialsTotal)
{
	// Ensure that agents and environments are of length numTrials
	if ((agents.size() != numTrialsTotal) || (environments.size() != numTrialsTotal))
		errorExit("Error in run(...). The number of agents/environments did not match numTrials.");

	int numExperiments = (int)numExperimentTrials.size();
	int idxFirstTrialNextExperiment = 0;
	vector<int> firstTrialNextExperiment;

    // Create a vector of matrices to store returns from all experiments; each matrix stores results from one experiment
	vector<MatrixXd> results(numExperiments);

	for (int experimentIdx = 0; experimentIdx < numExperiments; experimentIdx++)
	{
		int numEpisodes = maxEpisodes[idxFirstTrialNextExperiment];
		MatrixXd experimentResult(numExperimentTrials[experimentIdx], numEpisodes);
		results.push_back(experimentResult);
		idxFirstTrialNextExperiment += numExperimentTrials[experimentIdx];
	}

	int experimentIdx = 0;              // Index of result matrix in vector results to write G
	int trialCount = 0;
	
	// Create the object that we will return
	//MatrixXd result(numTrials, numEpisodes);

	// Each thread has its own random number generator, since generators are not thread safe
	vector<mt19937_64> generators(numTrialsTotal);
	for (int trial = 0; trial < numTrialsTotal; trial++)
		generators[trial].seed(generator());	// See with a random sample from the generator passed as an argument to this function

	cout << "\tThere are " << numTrialsTotal << " trials to run. Printing a * when each is completed..." << endl;

	// Loop over trials
#pragma omp parallel for	// This line instructs the compiler to parallelize the following for-loop. This uses openmp.
	for (int trial = 0; trial < numTrialsTotal; trial++)
	{
		// Run an agent lifetime
		double gamma = environments[trial]->getGamma(); // we have gammas vector?

		// Start change; why double?
		int numEpisodes = maxEpisodes[trial];
		//double numEpisodes = environments[trial]->getRecommendedMaxEpisodes;
		// End change

		// Loop over episodes
		for (int epCount = 0; epCount < numEpisodes; epCount++)
		{
			// Tell the agent and environment that we're starting a new episode
			agents[trial]->newEpisode(generators[trial]);
			environments[trial]->newEpisode(generators[trial]);

			if (agents[trial]->trainBeforeAPrime())
			{
				// Loop over time steps in this episode.
				// First define the variables we will use
				VectorXd curObs, newObs;
				int act;
				double reward, G = 0, curGamma = 1;

				// Get the initial observation
				environments[trial]->getObservation(generators[trial], curObs); // Writes the observation into curObs
				// Loop over time steps
				// Start change
				// End change
				for (int t = 0; t < maxEpisodeLength[trial]; t++) // After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
				{
					// Get action from the agent
					act = agents[trial]->getAction(curObs, generators[trial]);

					// Take the action, observe resulting reward
					reward = environments[trial]->step(act, generators[trial]);

					// Update the return
					G += curGamma * reward;

					// Update curGamma
					curGamma *= gamma;

					// Check if the episode is over
					if (environments[trial]->episodeOver(generators[trial])) {
						// Do a terminal update and break out of the loop over time
						agents[trial]->trainEpisodeEnd(curObs, act, reward, generators[trial]);
						break;
					}

					// Get the resulting observation
					environments[trial]->getObservation(generators[trial], newObs);

					// Train
					agents[trial]->train(curObs, act, reward, newObs, generators[trial]);

					// Copy new->cur
					curObs = newObs;
				}
				// Start change; save G to the correct place in the correct result matrix
				results[experimentIdx](trialCount, epCount) = G;
				// End change
				//result(trial, epCount) = G;
			}
			else
			{
				// Loop over time steps in this episode.
				// First define the variables we will use
				VectorXd curObs, newObs;
				int curAct, newAct;
				double reward, G = 0, curGamma = 1;

				// Get the initial observation and action
				environments[trial]->getObservation(generators[trial], curObs); // Writes the observation into curObs
				curAct = agents[trial]->getAction(curObs, generators[trial]);
				// Loop over time steps
				// Start change
				int maxEpiLen = maxEpisodeLength[trial];
				// End change

				for (int t = 0; t < maxEpiLen; t++) // After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
				{
					// Take the action, observe resulting reward
					reward = environments[trial]->step(curAct, generators[trial]);

					// Update the return
					G += curGamma * reward;

					// Update curGamma
					curGamma *= gamma;

					// Check if the episode is over
					if (environments[trial]->episodeOver(generators[trial]))
					{
						// Do a terminal update and break out of the loop over time
						agents[trial]->trainEpisodeEnd(curObs, curAct, reward, generators[trial]);
						break;
					}

					// Get the resulting observation
					environments[trial]->getObservation(generators[trial], newObs);

					// Get next action from the agent
					newAct = agents[trial]->getAction(newObs, generators[trial]);

					// Train
					agents[trial]->train(curObs, curAct, reward, newObs, newAct, generators[trial]);

					// Copy new->cur
					curAct = newAct;
					curObs = newObs;
				}
				// Start change; save G to the correct place in the correct result matrix
				results[experimentIdx](trialCount, epCount) = G;
				// End change
				// attempts to write next trial to array out of size
				//result(trial, epCount) = G;
			}
		}
		// Update indices
		if (trialCount > numExperimentTrials[experimentIdx])
		{
			experimentIdx += 1;
			trialCount = 0;
		}
		else if (experimentIdx < numExperiments)
			trialCount += 1;

		// End of a trial - print a star
		cout.put('*');
		cout.flush();
	}
	cout << endl; // We just printed a bunch of *'s. Put a newline so anything that prints after this starts on a new line.
	//return result;
	return results;
}

/*
#include <map>
int main() 
{
	unordered_map<string, double> my_map;
	my_map["apple"] = 5;
	my_map["banana"] = 3;

	for (auto& pair : my_map) 
	{
		cout << pair.first << ": " << pair.second << endl;
	}

	return 0;
}
*/

int main(int argc, char* argv[])
{
	// Comment out the line below if you don't want to run other random experiments first!
	//sandbox();

	// Default random number generator
	mt19937_64 generator;

	vector<int> numTrialsInExperiment;							// Length = total number of agent-environment pairs.   numTrialsInExperiment[i] is the number of trials to run for the i'th agent-environment pair.
	vector<string> agentNames;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
	vector<string> envNames;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
	vector<string> featureGenNames;								// Length = total number of trials (sum of elements in numTrialsInExperiment)																										
	vector<unordered_map<string, int>> featureGenParameters;	// Length = total number of trials (sum of elements in numTrialsInExperiment). i'th element is a map of parameters for this agent-environment pair's feature generator
	vector<unordered_map<string, double>> hyperParameters;		// Length = total number of trials (sum of elements in numTrialsInExperiment). i'th element is a map of hyperparameters for this agent-environment pair

	////////////////////////////////
	// Set parameters for experiments
	////////////////////////////////

	// Actor-Critic on Gridworld687
	numTrialsInExperiment.push_back(100);													// Set the number of trials for this algorithm (n)
	push_back_n((string)"Actor-Critic", numTrialsInExperiment.back(), agentNames);					// Add the agent name n times to the test vector (1D)
	push_back_n((string)"Gridworld687", numTrialsInExperiment.back(), envNames);					// Add the environment name n times to the test vector (1D)
	push_back_n((string)"Identity Basis", numTrialsInExperiment.back(), featureGenNames);
	push_back_n({ {"alpha", 0.001}, {"beta", 0.001}, {"lambda", 0.8} }, numTrialsInExperiment.back(), hyperParameters);

	// Actor-Critic on Mountain Car
	numTrialsInExperiment.push_back(100);
	push_back_n((string)"Sarsa(Lambda)", numTrialsInExperiment.back(), agentNames);
	push_back_n((string)"Mountain Car", numTrialsInExperiment.back(), envNames);
	push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);
	push_back_n({ {"alpha", 0.001}, {"beta", 0.001}, {"lambda", 0.8} }, numTrialsInExperiment.back(), hyperParameters);

	// Actor-Critic on Cart-Pole
	numTrialsInExperiment.push_back(100);
	push_back_n((string)"Actor-Critic", numTrialsInExperiment.back(), agentNames);
	push_back_n((string)"Cart-Pole", numTrialsInExperiment.back(), envNames);
	push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);
	push_back_n({ {"alpha", 0.001}, {"beta", 0.001}, {"lambda", 0.8} }, numTrialsInExperiment.back(), hyperParameters);

	// Sarsa(Lambda) on Mountain Car
	// Hyper parameters: {alphaSarsa, LambdaSarsa, EpsilonSarsa}
	numTrialsInExperiment.push_back(100);
	push_back_n((string)"Sarsa(Lambda)", numTrialsInExperiment.back(), agentNames);
	push_back_n((string)"Mountain Car", numTrialsInExperiment.back(), envNames);
	push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);
	push_back_n({ {"alpha", 0.001}, {"lambda", 0.8}, {"epsilon", 0.05} }, numTrialsInExperiment.back(), hyperParameters);

	// Sarsa(Lambda) on Cart-Pole
	numTrialsInExperiment.push_back(100);
	push_back_n((string)"Sarsa(Lambda)", numTrialsInExperiment.back(), agentNames);
	push_back_n((string)"Mountain Car", numTrialsInExperiment.back(), envNames);
	push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);
	push_back_n({ {"alpha", 0.001}, {"lambda", 0.8}, {"epsilon", 0.05} }, numTrialsInExperiment.back(), hyperParameters);

	// Q(Lambda) on Cart-Pole
	numTrialsInExperiment.push_back(100);
	push_back_n((string)"Q(Lambda)", numTrialsInExperiment.back(), agentNames);
	push_back_n((string)"Cart-Pole", numTrialsInExperiment.back(), envNames);
	push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);
	push_back_n({ {"alpha", 0.001}, {"lambda", 0.8}, {"epsilon", 0.05} }, numTrialsInExperiment.back(), hyperParameters);

	// Calculate the total number of trials
	int numTrialsTotal = 0;
	for (int i = 0; i < numTrialsInExperiment.size(); i++)
		numTrialsTotal += numTrialsInExperiment[i]

	////////////////////////////////
	// Create environment objects
	////////////////////////////////

	cout << "Creating environments..." << endl;
	vector<Environment*> environments(numTrialsTotal);
	for (int i = 0; i < numTrialsTotal; i++)
	{
		if (envNames[i] == "Gridworld")
			environments[i] = new Gridworld();
		else if (envNames[i] == "Gridworld687")
			environments[i] = new Gridworld687();
		else if (envNames[i] == "Mountain Car")
			environments[i] = new MountainCar();
		else if (envNames[i] == "Cart-Pole")
			environments[i] = new CartPole();
	}
	cout << "\tEnvironments created." << endl;

	////////////////////////////////
	// Get the parameters of each environment object
	////////////////////////////////

	vector<int> observationDimensions(numTrialsTotal);				// The dimension of the observation vector 
	vector<int> numActions(numTrialsTotal);                         // DO WE USE THIS? The number of actions
	VectorXi maxEpisodes(numTrialsTotal);							// The recommended episode length. Episodes might be terminated after this many time steps external to the environment (episodeOver will not necessarily return true).
	VectorXi maxEpisodeLengths(numTrialsTotal);						// The recommended maximum number of episodes for an agent lifetime for this environment.
	vector<double> gammas(numTrialsTotal);							// The discount factors
	vector<VectorXd> observationLowerBounds(numTrialsTotal);		// A lower bound on each observation feature
	vector<VectorXd> observationUpperBounds(numTrialsTotal);		// An upper bound on each observation feature

	for (int i = 0; i < numTrialsTotal; i++)
	{
		observationDimensions[i] = environments[i]->getObservationDimension();
		numActions[i] = environments[i]->getNumActions();
		maxEpisodes[i] = environments[i]->getRecommendedMaxEpisodes();
		maxEpisodeLengths[i] = environments[i]->getRecommendedEpisodeLength();
		observationLowerBounds[i] = environments[i]->getObservationLowerBound();
		observationUpperBounds[i] = environments[i]->getObservationUpperBound();
	}

	////////////////////////////////
	// Create feature generators 
	////////////////////////////////

	cout << "Creating feature generators..." << endl;
	vector<FeatureGenerator*> phis(numTrialsTotal);
	for (int trial = 0; trial < numTrialsTotal; trial++)
	{
		if (featureGenNames[trial] == "Fourier Basis")
			phis[trial] = new FourierBasis(observationDimensions[trial], observationLowerBounds[trial], observationUpperBounds[trial], featureGenParameters[trial]["iOrder"], featureGenParameters[trial]["dOrder"]);
		else if (featureGenNames[trial] == "Identity Basis")
			phis[trial] = new IdentityBasis(observationDimensions[trial], observationLowerBounds[trial], observationUpperBounds[trial]);
	}
	cout << "\tFeatures generators created." << endl;

	////////////////////////////////
	// Create agents
	////////////////////////////////

	// Now, actually create the agents
	cout << "Creating agents..." << endl;
	vector<Agent*> agents(numTrialsTotal);
	for (int trial = 0; trial < numTrialsTotal; trial++)
	{
		if (agentNames[trial] == "Actor-Critic")
			agents[trial] = new ActorCritic(observationDimensions[trial], numActions[trial], hyperParameters[trial]["alpha"], hyperParameters[trial]["beta"], hyperParameters[trial]["lambda"], gammas[trial], phis[trial]);
		else if (agentNames[trial] == "Sarsa(Lambda)")
			agents[trial] = new SarsaLambda(observationDimensions[trial], numActions[trial], hyperParameters[trial]["alpha"], hyperParameters[trial]["lambda"], hyperParameters[trial]["epsilon"], gammas[trial], phis[trial]);
		else if (agentNames[trial] == "Q(Lambda)")
			agents[trial] = new QLambda(observationDimensions[trial], numActions[trial], hyperParameters[trial]["alpha"], hyperParameters[trial]["lambda"], hyperParameters[trial]["epsilon"], gammas[trial], phis[trial]);
	}

	// Actually run the trials - this function is threaded!
	cout << "Running trials..." << endl;
	// Changed var type
	vector<MatrixXd> rawResults = run(agents, environments, maxEpisodes, maxEpisodeLengths, generator, numTrialsInExperiment, numTrialsTotal);
	//MatrixXd rawResults = run(agents, environments, numRunsTotal, maxEpisodes.maxCoeff(), maxEpisodeLength.maxCoeff(), generator);
	cout << "\tTrials completed." << endl;

	// Print the results to a file.
	cout << "Printing results to out/results.csv..." << endl;
#ifdef _MSC_VER	// Check if the compiler is a Microsoft compiler.
	//string filePath = "out/results.csv";	// If so, use this path
	string path = "out/results";	// If so, use this path
#else
	//string filePath = "../out/results.csv";	// Otherwise, use this path
	string path = "../out/results-" + to_string(numTrials) + " trials-";	// Otherwise, use this path
#endif
	// TO-DO: instead of using numSamples to write the results to CSV, write all results (so that CSV contains all results) and use numSaples parameter only for plotting
	int firstTrialInNextExperiment = 0;
	for (int experiment = 0; experiment < (int)numTrialsInExperiment.size(); experiment++)
	{
		int idx = firstTrialInNextExperiment + numTrialsInExperiment[experiment];
		string envName = envNames[idx];
		string agentFullName = agents[idx]->getName();
		string featureGenFullName = phis[idx]->getName();
		int maxEps = maxEpisodes[idx];

		string summaryFilePath = path + "summary-" + to_string(numTrialsInExperiment[idx]) + " trials-" + envName + featureGenFullName + agentFullName + ".csv";
		ofstream outResults(summaryFilePath);

		outResults << "Episode,Average Discounted Return,Standard Error" << endl;

		for (int epCount = 0; epCount < maxEps; epCount++)
		{
			double meanResult = rawResults[idx].col(epCount).mean();
			double standardError = sampleStandardError(rawResults[idx].col(epCount));

			outResults << epCount << "," << meanResult << "," << standardError << endl;
		}
	}

	// Clean up memory. Everything that we called "new" for, we need to call "delete" for.
	for (int i = 0; i < numTrialsTotal; i++)
	{
		delete environments[i];	// This call the deconstructor for environmnets[i], and then frees up the memory in the OS.
		delete phis[i];
		delete agents[i];
	}

	system("learning_curves.py");

	// Print message indicating that the program has finished
	cout << "Done. Press enter to exit." << endl;
	return 0; // No error.
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
// END V(-3)
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

// **************************************************************************************************

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
// START V(-2)
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

/*
* Run numTrials agent lifetimes on the provided environment. The entry in position (i,j) of the resulting matrix is the return from the j'th episode in the i'th trial.
*/

// Changed return type from MatrixXd to vector<MatrixXd>
// Added arguments int numEnvs, int numAlgs, int numRuns
vector<MatrixXd> run(vector<Agent*> agents, vector<Environment*> environments, int numTrials, VectorXi maxEpisodes, VectorXi maxEpisodeLength, mt19937_64& generator, int numEnvs, int numAlgs, int numRuns, int numRunsTotal)
{
	// Ensure that agents and environments are of length numTrials
	if ((agents.size() != numRunsTotal) || (environments.size() != numRunsTotal))
		errorExit("Error in run(...). The number of agents/environments did not match numTrials.");

	// Start change; create result matrix (numTrials, numEpisodes) for each environment-algorithm combination
	// Each result matrix will save G for episode in each trial for this environment-algorithm combination
	vector<MatrixXd> results;

	for (int i = 0; i < numEnvs * numAlgs; i++)
	{
		int numEpisodes = maxEpisodes[i * numTrials];
		MatrixXd result(numTrials, numEpisodes);
		results.push_back(result);
	}

	int resultIdx = 0;              // Index of result matrix in vector results to write G
	int trialResultIdxOffset = 0;   // Offset so that we can use [trial - trialResultIdxOffset] to write G into a correct place in one of the result matrices
	// End change

	// Create the object that we will return
	//MatrixXd result(numTrials, numEpisodes);

	// Each thread has its own random number generator, since generators are not thread safe
	vector<mt19937_64> generators(numRunsTotal);
	for (int i = 0; i < numRunsTotal; i++)
		generators[i].seed(generator());	// See with a random sample from the generator passed as an argument to this function

	cout << "\tThere are " << numRunsTotal << " trials to run. Printing a * when each is completed..." << endl;
	// Loop over trials
#pragma omp parallel for	// This line instructs the compiler to parallelize the following for-loop. This uses openmp.
	for (int trial = 0; trial < numRunsTotal; trial++)
	{

		// Start change; handle resultIdx and trialResultIdxOffset for trial
		if ((trial != 0) && (trial % numTrials == 0))
		{
			resultIdx += 1;                     // Increment resultIdx if moved to next algorithm (each env-alg is tested with numTrials trials)
			trialResultIdxOffset += numTrials;  // Increment trialResultIdxOffset
		}
		// End change

		// Run an agent lifetime
		double gamma = environments[trial]->getGamma();

		// Start change; why double?
		int numEpisodes = maxEpisodes[trial];
		//double numEpisodes = environments[trial]->getRecommendedMaxEpisodes;
		// End change

		// Loop over episodes
		for (int epCount = 0; epCount < numEpisodes; epCount++)
		{
			// Tell the agent and environment that we're starting a new episode
			agents[trial]->newEpisode(generators[trial]);
			environments[trial]->newEpisode(generators[trial]);

			if (agents[trial]->trainBeforeAPrime())
			{
				// Loop over time steps in this episode.
				// First define the variables we will use
				VectorXd curObs, newObs;
				int act;
				double reward, G = 0, curGamma = 1;

				// Get the initial observation
				environments[trial]->getObservation(generators[trial], curObs); // Writes the observation into curObs
				// Loop over time steps
				// Start change
				int maxEpiLen = maxEpisodeLength[trial];
				// End change
				for (int t = 0; t < maxEpiLen; t++) // After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
				{
					// Get action from the agent
					act = agents[trial]->getAction(curObs, generators[trial]);

					// Take the action, observe resulting reward
					reward = environments[trial]->step(act, generators[trial]);

					// Update the return
					G += curGamma * reward;

					// Update curGamma
					curGamma *= gamma;

					// Check if the episode is over
					if (environments[trial]->episodeOver(generators[trial])) {
						// Do a terminal update and break out of the loop over time
						agents[trial]->trainEpisodeEnd(curObs, act, reward, generators[trial]);
						break;
					}

					// Get the resulting observation
					environments[trial]->getObservation(generators[trial], newObs);

					// Train
					agents[trial]->train(curObs, act, reward, newObs, generators[trial]);

					// Copy new->cur
					curObs = newObs;
				}
				// Start change; save G to the correct place in the correct result matrix
				results[resultIdx]((trial - trialResultIdxOffset), epCount) = G;
				// End change
				//result(trial, epCount) = G;
			}
			else
			{
				// Loop over time steps in this episode.
				// First define the variables we will use
				VectorXd curObs, newObs;
				int curAct, newAct;
				double reward, G = 0, curGamma = 1;

				// Get the initial observation and action
				environments[trial]->getObservation(generators[trial], curObs); // Writes the observation into curObs
				curAct = agents[trial]->getAction(curObs, generators[trial]);
				// Loop over time steps
				// Start change
				int maxEpiLen = maxEpisodeLength[trial];
				// End change

				for (int t = 0; t < maxEpiLen; t++) // After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
				{
					// Take the action, observe resulting reward
					reward = environments[trial]->step(curAct, generators[trial]);

					// Update the return
					G += curGamma * reward;

					// Update curGamma
					curGamma *= gamma;

					// Check if the episode is over
					if (environments[trial]->episodeOver(generators[trial]))
					{
						// Do a terminal update and break out of the loop over time
						agents[trial]->trainEpisodeEnd(curObs, curAct, reward, generators[trial]);
						break;
					}

					// Get the resulting observation
					environments[trial]->getObservation(generators[trial], newObs);

					// Get next action from the agent
					newAct = agents[trial]->getAction(newObs, generators[trial]);

					// Train
					agents[trial]->train(curObs, curAct, reward, newObs, newAct, generators[trial]);

					// Copy new->cur
					curAct = newAct;
					curObs = newObs;
				}
				// Start change; save G to the correct place in the correct result matrix
				results[resultIdx]((trial - trialResultIdxOffset), epCount) = G;
				// End change
				// attempts to write next trial to array out of size
				//result(trial, epCount) = G;
			}
		}
		// End of a trial - print a star
		cout.put('*');
		cout.flush();
	}
	cout << endl; // We just printed a bunch of *'s. Put a newline so anything that prints after this starts on a new line.
	//return result;
	return results;
}

// TODO: I think we should delete the function below once the one above is set up properly. It is similar, but threaded!

// Takes as input an agent and an environment, runs the agent on the environment for one agent lifetime.
// Returns the vector of returns from each episode.
VectorXd runLifetime(Agent* agent, Environment* env, int numEpisodes, int maxEpisodeLength,
	mt19937_64& generator, VectorXd& totalTimeStepsAtEpisodeEndsBuff)	// Buff means "buffer" - we are going to return values in this object.
{
	VectorXd result(numEpisodes);	// Create the array where we will return the resulting returns (G).
	totalTimeStepsAtEpisodeEndsBuff.resize(numEpisodes);
	VectorXd episodeLengths(numEpisodes); // Create the array where we will store the total number of steps taken across all episodes up to that moment in time.
	double gamma = env->getGamma();
	int actCount = 0;

	// Loop over episodes
	for (int epCount = 0; epCount < numEpisodes; epCount++)
	{
		// Tell the agent and environment that we're starting a new episode
		agent->newEpisode(generator);
		env->newEpisode(generator);

		if (agent->trainBeforeAPrime())
		{
			// Loop over time steps in this episode.
			// First define the variables we will use
			VectorXd curObs, newObs;
			int act;
			double reward, G = 0, curGamma = 1;

			// Get the initial observation
			env->getObservation(generator, curObs); // Writes the observation into curObs
			// Loop over time steps
			for (int t = 0; t < maxEpisodeLength; t++) // After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
			{
				// Get action from the agent
				act = agent->getAction(curObs, generator);
				actCount++; // Increment action count

				// Take the action, observe resulting reward
				reward = env->step(act, generator);

				// Update the return
				G += curGamma * reward;

				// Update curGamma
				curGamma *= gamma;

				// Check if the episode is over
				if (env->episodeOver(generator))
				{
					// Do a terminal update and break out of the loop over time
					agent->trainEpisodeEnd(curObs, act, reward, generator);
					break;
				}

				// Get the resulting observation
				env->getObservation(generator, newObs);

				// Train
				agent->train(curObs, act, reward, newObs, generator);

				// Copy new->cur
				curObs = newObs;
			}
			result[epCount] = G;
			totalTimeStepsAtEpisodeEndsBuff[epCount] = actCount;
		}
		else
		{
			// Loop over time steps in this episode.
			// First define the variables we will use
			VectorXd curObs, newObs;
			int curAct, newAct;
			double reward, G = 0, curGamma = 1;

			// Get the initial observation and action
			env->getObservation(generator, curObs); // Writes the observation into curObs
			curAct = agent->getAction(curObs, generator);
			actCount++; // Increment action count
			// Loop over time steps
			for (int t = 0; t < maxEpisodeLength; t++) // After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
			{
				// Take the action, observe resulting reward
				reward = env->step(curAct, generator);

				// Update the return
				G += curGamma * reward;

				// Update curGamma
				curGamma *= gamma;

				// Check if the episode is over
				if (env->episodeOver(generator))
				{
					// Do a terminal update and break out of the loop over time
					agent->trainEpisodeEnd(curObs, curAct, reward, generator);
					break;
				}

				// Get the resulting observation
				env->getObservation(generator, newObs);

				// Get next action from the agent
				newAct = agent->getAction(newObs, generator);
				actCount++; // Increment action count

				// Train
				agent->train(curObs, curAct, reward, newObs, newAct, generator);

				// Copy new->cur
				curAct = newAct;
				curObs = newObs;
			}
			result[epCount] = G;
			totalTimeStepsAtEpisodeEndsBuff[epCount] = actCount;
		}
	}

	// Return the results that we computed
	return result;
}

void sandbox()
{
	cout << "Running other experiments/code." << endl;
	//sandboxJune16_2023();
	//sandboxJune16_2023();

	mt19937_64 generator;
	VectorXd observation = VectorXd::Zero(4);
	while (true)
	{
		VectorXd inputLowerBound = VectorXd::Zero(4);
		VectorXd inputUpperBound = VectorXd::Ones(4);
		FourierBasis phi(4, inputLowerBound, inputUpperBound, 3, 3);
		SarsaLambda s(10, 5, 0.01, .8, .1, 1.0, &phi);
		cout << s.getAction(observation, generator) << endl;
	}


	cout << "Done running other experiments/code. Hit enter to continue." << endl;
	(void)getchar();
}

// Here is the old main! Ith as hyperparameter settings you may have been working with.
/*
int main(int argc, char* argv[])
{
	// Comment out the line below if you don't want to run other random experiments first!
//	sandbox();

	// Set hyperparameters and RNG
	//double  alpha = 0.001, beta = 0.001, lambda = 0.9, epsilon = 0.0, sigma = 0.5; // mc
	//double  alpha = 0.0001, beta = 0.0001, lambda = 0.8; // Phil mc

	double  alpha = 0.0001, beta = 0.0001, lambda = 0.8; // Phil mc 2

//    double  alpha = 0.001, lambda = 0.9, epsilon = 0.0;
//	double  alpha = 0.01, beta = 0.01, lambda = 0.8, epsilon = 0.05, sigma = 0.1; // Qlambda, Actor-Critic, gr687
//	double  alpha = 0.1, beta = 0.1, lambda = 0.8, epsilon = 0.05, sigma = 0.1;

	//int iOrder = 5, dOrder = 5; // mc 5, 5
	//int iOrder = 3, dOrder = 3; // Phil's attempt at MC
	int iOrder = 7, dOrder = 7; // Phil's attempt at MC 2
	mt19937_64 generator;	// If you don't seed it, it has some fixed seed that is the same every time.

	// Create the environment
//	Gridworld env(4);
//  Gridworld687 env;
//  MountainCar env;
	MountainCar env(false);

	// Get parameters of the environment
	int observationDimension = env.getObservationDimension(), numActions = env.getNumActions(),
		maxEpisodes = env.getRecommendedMaxEpisodes(), maxEpisodeLength = env.getRecommendedEpisodeLength();
	double gamma = env.getGamma();
	VectorXd observationLowerBound = env.getObservationLowerBound(),
		observationUpperBound = env.getObservationUpperBound();

	// Create the FourierBasis object
	FourierBasis phi(observationDimension, observationLowerBound, observationUpperBound, iOrder, dOrder);

	// Create the agent
//	SarsaLambda agent(observationDimension, numActions, alpha, lambda, epsilon, gamma, &phi); // The &phi means "the memory location of phi". Notice the constructor takes a pointer FeatureGenerator*.
//    QLambda agent(observationDimension, numActions, alpha, lambda, epsilon, gamma, &phi); // The &phi means "the memory location of phi". Notice the constructor takes a pointer FeatureGenerator*.
	ActorCritic agent(observationDimension, numActions, alpha, beta, lambda, gamma, &phi); // The &phi means "the memory location of phi". Notice the constructor takes a pointer FeatureGenerator*.

	// Run the agent on the environment
	VectorXd totalTimeStepsAtEpisodeEnds;	// runLifetime will store the total number of timesteps that have passed when each episode ends. It will store it in this array.
	VectorXd returns = runLifetime(&agent, &env, maxEpisodes, maxEpisodeLength, generator, totalTimeStepsAtEpisodeEnds);

	// Print totalTimeStepsAtEpisodeEnds to file

#ifdef _MSC_VER	// Check if the compiler is a Microsoft compiler.
	string filePath = "out/totalTimeStepsAtEpisodeEnds.txt";	// If so, use this path
#else
	string filePath = "../out/totalTimeStepsAtEpisodeEnds.txt";	// Otherwise, use this path
#endif
	ofstream outTimeSteps(filePath);
	outTimeSteps << fixed << totalTimeStepsAtEpisodeEnds << endl;
	outTimeSteps.close();

	// Print the returns
	cout << "Returns:" << endl << returns << endl;

	// Print the returns to a file
#ifdef _MSC_VER	// Check if the compiler is a Microsoft compiler.
	filePath = "out/returns.txt";	// If so, use this path
#else
	filePath = "../out/returns.txt";	// Otherwise, use this path
#endif
	ofstream outReturns(filePath);
	outReturns << returns << endl;
	outReturns.close();

	// Print message indicating that the program has finished
	cout << "Done. Press enter to exit." << endl;
	return 0; // No error.
}
*/

int main(int argc, char* argv[])
{
	// Comment out the line below if you don't want to run other random experiments first!
	//sandbox();

	// Default random number generator
	mt19937_64 generator;

	// Hyperparameters
	int numTrials = 2, numAlgs = 3, numEnvs = 3;
	int numRuns = numTrials * numAlgs, numRunsTotal = numRuns * numEnvs;		// DONE! @TODO: Reverse variable names. numTrials is the number of times each algorithm/environmnet pair is run, numRuns is the total number of runs that will happen.
	int numSamples = 1; // How many samples (average) we use for the plot

	// Environments
	int iOrderGridworld687 = 0, dOrderGridworld687 = 0;
	int iOrderMountainCar = 3, dOrderMountainCar = 3;
	int iOrderCartPole = 3, dOrderCartPole = 3;

	// Agents
	double alphaAC = 0.0001, betaAC = 0.0001, lambdaAC = 0.8;
	double alphaSarsa = 0.0001, lambdaSarsa = 0.8, epsilonSarsa = 0.01;
	double alphaQ = 0.0001, lambdaQ = 0.8, epsilonQ = 0.01;
	double alphaExpectedSarsa = 0.0001, lambdaExpectedSarsa = 0.8, epsilonExpectedSarsa = 0.01;
	double alphaReinforce = 0.01;

	// Create the environment objects
	cout << "Creating environments..." << endl;
	vector<Environment*> environments(numRunsTotal);
	for (int i = 0; i < numRuns; i++)
	{
		environments[i] = new Gridworld687();
		environments[i + numRuns] = new MountainCar();
		environments[i + 2 * numRuns] = new CartPole();
	}
	cout << "\tEnvironments created." << endl;

	// Get parameters of the environment
	vector<string> environmentName(numRunsTotal);
	vector<int> observationDimension(numRunsTotal), numActions(numRunsTotal);
	VectorXi maxEpisodes(numRunsTotal), maxEpisodeLength(numRunsTotal);
	vector<double> gamma(numRunsTotal);
	vector<VectorXd> observationLowerBound(numRunsTotal), observationUpperBound(numRunsTotal);

	for (int i = 0; i < numRunsTotal; i++)
	{
		environmentName[i] = environments[i]->getName();
		observationDimension[i] = environments[i]->getObservationDimension();
		numActions[i] = environments[i]->getNumActions();
		maxEpisodes[i] = environments[i]->getRecommendedMaxEpisodes();
		maxEpisodeLength[i] = environments[i]->getRecommendedEpisodeLength();
		observationLowerBound[i] = environments[i]->getObservationLowerBound();
		observationUpperBound[i] = environments[i]->getObservationUpperBound();
	}


	// Get parameters for the feature generator
	vector<int> iOrder(numRunsTotal), dOrder(numRunsTotal);
	for (int i = 0; i < numRunsTotal; i++)
	{
		if (environmentName[i] == "Gridworld687")
		{
			iOrder[i] = iOrderGridworld687;
			dOrder[i] = dOrderGridworld687;
		}
		else if (environmentName[i] == "Mountain Car")
		{
			iOrder[i] = iOrderMountainCar;
			dOrder[i] = dOrderMountainCar;
		}
		else if (environmentName[i] == "Cart-Pole")
		{
			iOrder[i] = iOrderCartPole;
			dOrder[i] = dOrderCartPole;
		}
	}

	// Create agents. First, we need the FeatureGenerator objects - one for each!
	cout << "Creating feature generators..." << endl;
	vector<FeatureGenerator*> phis(numRunsTotal);
	for (int i = 0; i < numRunsTotal; i++)
		phis[i] = new FourierBasis(observationDimension[i], observationLowerBound[i], observationUpperBound[i], iOrder[i], dOrder[i]);
	cout << "\tFeatures generators created." << endl;

	// Now, actually create the agents
	cout << "Creating agents..." << endl;
	vector<Agent*> agents(numRunsTotal);
	for (int i = 0; i < numRunsTotal; i += numRuns)
	{
		for (int j = 0; j < numTrials; j++)
		{
			int idx = i + j;
			agents[idx] = new ActorCritic(observationDimension[idx], numActions[idx], alphaAC, betaAC, lambdaAC, gamma[idx], phis[idx]);
			agents[idx + numTrials] = new SarsaLambda(observationDimension[idx + numTrials], numActions[idx + numTrials], alphaSarsa, lambdaSarsa, epsilonSarsa, gamma[idx + numTrials], phis[idx + numTrials]);
			agents[idx + 2 * numTrials] = new QLambda(observationDimension[idx + 2 * numTrials], numActions[idx + 2 * numTrials], alphaQ, lambdaQ, epsilonQ, gamma[idx + 2 * numTrials], phis[idx + 2 * numTrials]);
			//agents[idx + 3 * numTrials] = new ExpectedSarsaLambda(observationDimension[idx + 3 * numTrials], numActions[idx + 3 * numTrials], alphaQ, lambdaQ, epsilonQ, gamma[idx + 3 * numTrials], phis[idx + 3 * numTrials]);
			//agents[idx + 4 * numTrials] = new Reinforce(observationDimension[idx + 4 * numTrials], numActions[idx + 4 * numTrials], alphaReinforce, gamma[idx + 4 * numTrials], phis[idx + 4 * numTrials]);
		}
	}

	cout << "\tAgents created." << endl;

	// Actually run the trials - this function is threaded!
	cout << "Running trials..." << endl;
	// Changed var type
	vector<MatrixXd> rawResults = run(agents, environments, numTrials, maxEpisodes, maxEpisodeLength, generator, numEnvs, numAlgs, numRuns, numRunsTotal);
	//MatrixXd rawResults = run(agents, environments, numRunsTotal, maxEpisodes.maxCoeff(), maxEpisodeLength.maxCoeff(), generator);
	cout << "\tTrials completed." << endl;

	// Print the results to a file.
	cout << "Printing results to out/results.csv..." << endl;
#ifdef _MSC_VER	// Check if the compiler is a Microsoft compiler.
	//string filePath = "out/results.csv";	// If so, use this path
	string path = "out/results-" + to_string(numTrials) + " trials-";	// If so, use this path
#else
	//string filePath = "../out/results.csv";	// Otherwise, use this path
	string path = "../out/results-" + to_string(numTrials) + " trials-";	// Otherwise, use this path
#endif
	// TO-DO: instead of using numSamples to write the results to CSV, write all results (so that CSV contains all results) and use numSaples parameter only for plotting
	for (int i = 0; i < numEnvs * numAlgs; i++)
	{
		int idx = i * numTrials;
		string envName = environmentName[idx];
		string algName = agents[idx]->getName();
		int maxEps = maxEpisodes[idx];

		string filePath = path + envName + " with iO " + to_string(iOrder[idx]) + ", dO " + to_string(dOrder[idx]) + "-" + algName + ".csv";
		ofstream outResults(filePath);

		double meanResult = 0, meanStandardError = 0;

		outResults << "Episode,Average Discounted Return,Standard Error" << endl;

		for (int epCount = 0; epCount < maxEps; epCount++)
		{
			meanResult += rawResults[i].col(epCount).mean();
			meanStandardError += sampleStandardError(rawResults[i].col(epCount));
			if ((epCount + 1) % numSamples == 0)
			{
				outResults << epCount << "," << meanResult / (double)numSamples << "," << meanStandardError / (double)numSamples << endl;	// The functions 'sampleMean' and 'sampleStandardError' are defined in common.hpp
				meanResult = meanStandardError = 0;
			}
		}
		outResults.close();
	}
	cout << "\tResults printed." << endl;

	// Clean up memory. Everything that we called "new" for, we need to call "delete" for.
	for (int i = 0; i < numRuns; i++)
	{
		delete environments[i];	// This call the deconstructor for environmnets[i], and then frees up the memory in the OS.
		delete phis[i];
		delete agents[i];
	}

	system("learning_curves.py");

	// Print message indicating that the program has finished
	cout << "Done. Press enter to exit." << endl;
	return 0; // No error.
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
// END V(-2)
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

// **************************************************************************************************