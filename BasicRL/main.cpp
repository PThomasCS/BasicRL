/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
// START V(-3)
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.hpp"
#include <unordered_map>
#include <iomanip> 

using namespace std;
using namespace Eigen;

// Run numTrials agent lifetimes on the provided environment. The entry in position (i,j) of the resulting matrix is the return from the j'th episode in the i'th trial.

vector<MatrixXd> run(vector<Agent*> agents, vector<Environment*> environments, VectorXi maxEpisodes, VectorXi maxEpisodeLengths, mt19937_64& generator, 
	vector<int> numExperimentTrials, int numTrialsTotal, vector<int> experimentIDs, vector<int> trialCounts)
{
	// Ensure that agents and environments are of length numTrialsTotal
	if ((agents.size() != numTrialsTotal) || (environments.size() != numTrialsTotal))
		errorExit("Error in run(...). The number of agents/environments did not match numTrials.");

	int numExperiments = (int)numExperimentTrials.size();   // The total number of experiments
	int idxFirstTrialInExperiment = 0;                      // The index of the first trial in the current experiment

	// Create a vector of matrices to store returns from all experiments; each matrix stores returns from one experiment
	vector<MatrixXd> results;                               // Length = the total number of experiments

	cout << "numExperiments: " << numExperiments << endl;

	for (int experimentIdx = 0; experimentIdx < numExperiments; experimentIdx++)
	{   
		int numEpisodes = maxEpisodes[idxFirstTrialInExperiment];                           // The maximum number of episodes for the environment used in an experiment
		MatrixXd experimentResult(numExperimentTrials[experimentIdx], numEpisodes);         // The matrix to store returns from an experiment; rows = number of trials in the experiment, cols = maximum number of episodes
		experimentResult.setConstant(-24623467); // @TODO: Alexandra, debug from here then remove this!
		results.push_back(experimentResult);
		idxFirstTrialInExperiment += numExperimentTrials[experimentIdx];                     // Update the index of the first trial in an experiment (to get the correct numEpisodes for the next experiment)
	}

	//int experimentIdx = 0;      // The ndex of the result matrix in the vector of results
	//int trialCount = 0;         // The index for a trial in an experiment

	// Each thread has its own random number generator, since generators are not thread safe
	vector<mt19937_64> generators(numTrialsTotal);
	for (int trial = 0; trial < numTrialsTotal; trial++)
		generators[trial].seed(generator());	// Seed with a random sample from the generator passed as an argument to this function

	cout << "\tThere are " << numTrialsTotal << " trials to run. Printing a * when each is completed..." << endl;

	// Loop over trials
#pragma omp parallel for	// This line instructs the compiler to parallelize the following for-loop. This uses openmp.
	for (int trial = 0; trial < numTrialsTotal; trial++)
	{
		// Run an agent lifetime
		double gamma = environments[trial]->getGamma(); // Or pass the gammas vector as an argument and use it?

		int numEpisodes = maxEpisodes[trial];           // Why did we use doubles?
		//double numEpisodes = environments[trial]->getRecommendedMaxEpisodes;

		// Loop over episodes in trial
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
				//  After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
				for (int t = 0; t < maxEpisodeLengths[trial]; t++)
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
					if (environments[trial]->episodeOver(generators[trial]))
					{
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
				// End of episode
				// Save G to the correct place in the correct result matrix
				//results[experimentIdx](trialCount, epCount) = G;
				results[experimentIDs[trial]](trialCounts[trial], epCount) = G;
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
				int maxEpiLen = maxEpisodeLengths[trial];
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
				// End of episode
				// Save G to the correct place in the correct result matrix
				//results[experimentIdx](trialCount, epCount) = G;
				results[experimentIDs[trial]](trialCounts[trial], epCount) = G;
			}
		}
		// End of trial
		//// Update indices
		//if (trialCount == (numExperimentTrials[experimentIdx] - 1))
		//{
		//	experimentIdx += 1; // In the last trial we update this to be out of range (it's fine)
		//	trialCount = 0;
		//}
		//else
		//	trialCount += 1;

		// End of a trial - print a star
		cout.put('*');
		cout.flush();
	}
	cout << endl; // We just printed a bunch of *'s. Put a newline so anything that prints after this starts on a new line.
	//return result;
	return results;
}

int main(int argc, char* argv[])
{
	// Comment out the line below if you don't want to run other random experiments first!
	//sandbox();
    mt19937_64 g;

    ofstream outAlpha("out/alpha_sample.txt");
    for (int i = 0; i < 100; i++)
        outAlpha << sampleParameter("alpha", g) << endl;
    outAlpha.close();

    ofstream outBeta("out/beta_sample.txt");
    for (int i = 0; i < 100; i++)
        outBeta << sampleParameter("beta", g) << endl;
    outBeta.close();

    ofstream outEpsilon("out/epsilon_sample.txt");
    for (int i = 0; i < 100; i++)
        outEpsilon << sampleParameter("epsilon", g) << endl;
    outEpsilon.close();

    ofstream outLambda("out/lambda_sample.txt");
    for (int i = 0; i < 100; i++)
        outLambda << sampleParameter("lambda", g) << endl;
    outLambda.close();

    system("python sandbox_plots.py");

    // Default random number generator
	mt19937_64 generator;

	vector<int> numHyperParamExperiments;                       // Length = total number of sets of hyperparameters
	vector<int> numTrialsInExperiment;							// Length = total number of agent-environment pairs.   numTrialsInExperiment[i] is the number of trials to run for the i'th agent-environment pair.
	vector<int> experimentIDs;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
	vector<int> trialCounts;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
	vector<string> agentNames;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
	vector<string> envNames;									// Length = total number of trials (sum of elements in numTrialsInExperiment)
	vector<string> featureGenNames;								// Length = total number of trials (sum of elements in numTrialsInExperiment)																										
	vector<unordered_map<string, int>> featureGenParameters;	// Length = total number of trials (sum of elements in numTrialsInExperiment). i'th element is a map of parameters for this agent-environment pair's feature generator
	vector<unordered_map<string, double>> hyperParameters;		// Length = total number of trials (sum of elements in numTrialsInExperiment). i'th element is a map of hyperparameters for this agent-environment pair

	////////////////////////////////
	// Set parameters for experiments
	////////////////////////////////

    // Sarsa-Lambda on Gridworld687
	numHyperParamExperiments.push_back(10);
	for (int hyperParamExp = 0; hyperParamExp < numHyperParamExperiments.back(); hyperParamExp++)
	{
		numTrialsInExperiment.push_back(2);
		push_back_0_n(numTrialsInExperiment.back(), trialCounts);
		push_back_n((string)"Sarsa(Lambda)", numTrialsInExperiment.back(), agentNames);
		push_back_n((string)"Gridworld687", numTrialsInExperiment.back(), envNames);
		push_back_n((string)"Fourier Basis", numTrialsInExperiment.back(), featureGenNames);
		push_back_n({ {"iOrder", 0}, {"dOrder", 0} }, numTrialsInExperiment.back(), featureGenParameters);

		if (hyperParamExp < 1)
		{
			push_back_n({ {"alpha", 0.01}, {"lambda", 0.8}, {"epsilon", 0.05} }, numTrialsInExperiment.back(), hyperParameters);
		}
		else
		{
			push_back_n({ {"alpha", sampleHyperParameter((string)"alpha", generator)}, {"lambda", sampleHyperParameter((string)"lambda", generator)},
				{"epsilon", sampleHyperParameter((string)"epsilon", generator)} }, numTrialsInExperiment.back(), hyperParameters);
		}
	}

    // Q(lambda) on Mountain Car
    numHyperParamExperiments.push_back(2);
	for (int hyperParamExp = 0; hyperParamExp < numHyperParamExperiments.back(); hyperParamExp++)
	{
		numTrialsInExperiment.push_back(2);
		push_back_0_n(numTrialsInExperiment.back(), trialCounts);
		push_back_n((string)"Q(Lambda)", numTrialsInExperiment.back(), agentNames);
		push_back_n((string)"Mountain Car", numTrialsInExperiment.back(), envNames);
		push_back_n((string)"Fourier Basis", numTrialsInExperiment.back(), featureGenNames);
		push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);

		if (hyperParamExp < 1)
		{
			push_back_n({ {"alpha", 0.0001}, {"lambda", 0.8}, {"epsilon", 0.05} }, numTrialsInExperiment.back(), hyperParameters);
		}
		else
		{
			push_back_n({ {"alpha", sampleHyperParameter((string)"alpha", generator)}, {"lambda", sampleHyperParameter((string)"lambda", generator)},
				{"epsilon", sampleHyperParameter((string)"epsilon", generator)} }, numTrialsInExperiment.back(), hyperParameters);
		}
	}

	// Actor-Critic on Cart-Pole
	numHyperParamExperiments.push_back(2);
	for (int hyperParamExp = 0; hyperParamExp < numHyperParamExperiments.back(); hyperParamExp++)
	{
		numTrialsInExperiment.push_back(2);
		push_back_0_n(numTrialsInExperiment.back(), trialCounts);
		push_back_n((string)"Actor-Critic", numTrialsInExperiment.back(), agentNames);
		push_back_n((string)"Cart-Pole", numTrialsInExperiment.back(), envNames);
		push_back_n((string)"Fourier Basis", numTrialsInExperiment.back(), featureGenNames);
		push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);

		if (hyperParamExp < 1)
		{
			push_back_n({ {"alpha", 0.001}, {"beta", 0.001}, {"lambda", 0.8} }, numTrialsInExperiment.back(), hyperParameters);
		}
		else
		{
			push_back_n({ {"alpha", sampleHyperParameter((string)"alpha", generator)}, {"beta", sampleHyperParameter((string)"beta", generator)},
				{"lambda", sampleHyperParameter((string)"lambda", generator)} }, numTrialsInExperiment.back(), hyperParameters);
		}
	}
//
//    // Expected Sarsa on Mountain Car
//	numHyperParamExperiments.push_back(2);
//	for (int hyperParamExp = 0; hyperParamExp < numHyperParamExperiments.back(); hyperParamExp++)
//	{
//		numTrialsInExperiment.push_back(2);
//		push_back_n((string)"Expected Sarsa (Lambda)", numTrialsInExperiment.back(), agentNames);
//		push_back_n((string)"Mountain Car", numTrialsInExperiment.back(), envNames);
//		push_back_n((string)"Fourier Basis", numTrialsInExperiment.back(), featureGenNames);
//		push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);
//
//		if (hyperParamExp < 1)
//		{
//			push_back_n({ {"alpha", 0.001}, {"lambda", 0.8}, {"epsilon", 0.05} }, numTrialsInExperiment.back(), hyperParameters);
//		}
//		else
//		{
//			push_back_n({ {"alpha", sampleHyperParameter((string)"alpha", generator)}, {"lambda", sampleHyperParameter((string)"lambda", generator)},
//				{"epsilon", sampleHyperParameter((string)"epsilon", generator)} }, numTrialsInExperiment.back(), hyperParameters);
//		}
//	}
//
//    // Reinforce on Cart-Pole
//	numHyperParamExperiments.push_back(2);
//	for (int hyperParamExp = 0; hyperParamExp < numHyperParamExperiments.back(); hyperParamExp++)
//	{
//		numTrialsInExperiment.push_back(2);
//		push_back_n((string)"REINFORCE", numTrialsInExperiment.back(), agentNames);
//		push_back_n((string)"Cart-Pole", numTrialsInExperiment.back(), envNames);
//		push_back_n((string)"Fourier Basis", numTrialsInExperiment.back(), featureGenNames);
//		push_back_n({ {"iOrder", 3}, {"dOrder", 3} }, numTrialsInExperiment.back(), featureGenParameters);
//
//		if (hyperParamExp < 1)
//		{
//			push_back_n({ {"alpha", 0.001}, numTrialsInExperiment.back(), hyperParameters);
//		}
//		else
//		{
//			push_back_n({ {"alpha", sampleHyperParameter((string)"alpha", generator)} }, numTrialsInExperiment.back(), hyperParameters);
//		}
//	}

	// Calculate the total number of trials
	int numTrialsTotal = 0;
	for (int experiment = 0; experiment < numTrialsInExperiment.size(); experiment++)
		numTrialsTotal += numTrialsInExperiment[experiment];

	// Initialize the vector contating experimentIDs
	for (int experimentID = 0; experimentID < numTrialsInExperiment.size(); experimentID++)
	{
		push_back_n(experimentID, numTrialsInExperiment[experimentID], experimentIDs);
	}

	///////////////////////////////////////////////////
	// Create environment objects
	///////////////////////////////////////////////////

	cout << "Creating environments..." << endl;
	vector<Environment*> environments(numTrialsTotal);
	for (int trial = 0; trial < numTrialsTotal; trial++)
	{
		if (envNames[trial] == "Gridworld")
			environments[trial] = new Gridworld();
		else if (envNames[trial] == "Gridworld687")
			environments[trial] = new Gridworld687();
		else if (envNames[trial] == "Mountain Car")
			environments[trial] = new MountainCar();
		else if (envNames[trial] == "Cart-Pole")
			environments[trial] = new CartPole();
	}
	cout << "\tEnvironments created." << endl;

	///////////////////////////////////////////////////
	// Get the parameters of each environment object
	///////////////////////////////////////////////////

	vector<int> observationDimensions(numTrialsTotal);				// The dimension of the observation vector 
	vector<int> numActions(numTrialsTotal);                         // DO WE USE THIS? The number of actions
	VectorXi maxEpisodes(numTrialsTotal);							// The recommended episode length. Episodes might be terminated after this many time steps external to the environment (episodeOver will not necessarily return true).
	VectorXi maxEpisodeLengths(numTrialsTotal);						// The recommended maximum number of episodes for an agent lifetime for this environment.
	vector<double> gammas(numTrialsTotal);							// The discount factors
	vector<VectorXd> observationLowerBounds(numTrialsTotal);		// A lower bound on each observation feature
	vector<VectorXd> observationUpperBounds(numTrialsTotal);		// An upper bound on each observation feature

	for (int trial = 0; trial < numTrialsTotal; trial++)
	{
		observationDimensions[trial] = environments[trial]->getObservationDimension();
		numActions[trial] = environments[trial]->getNumActions();
		maxEpisodes[trial] = environments[trial]->getRecommendedMaxEpisodes();
		maxEpisodeLengths[trial] = environments[trial]->getRecommendedEpisodeLength();
		observationLowerBounds[trial] = environments[trial]->getObservationLowerBound();
		observationUpperBounds[trial] = environments[trial]->getObservationUpperBound();
		gammas[trial] = environments[trial]->getGamma();
	}

	///////////////////////////////////////////////////
	// Create feature generators 
	///////////////////////////////////////////////////

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

	///////////////////////////////////////////////////
	// Create agents
	///////////////////////////////////////////////////

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
        else if (agentNames[trial] == "Experted Sarsa(Lambda)")
            agents[trial] = new ExpectedSarsaLambda(observationDimensions[trial], numActions[trial], hyperParameters[trial]["alpha"], hyperParameters[trial]["lambda"], hyperParameters[trial]["epsilon"], gammas[trial], phis[trial]);
        else if (agentNames[trial] == "REINFORCE")
            agents[trial] = new Reinforce(observationDimensions[trial], numActions[trial], hyperParameters[trial]["alpha"], gammas[trial], phis[trial]);
	}

	// Actually run the trials - this function is threaded!
	cout << "Running trials..." << endl;
	// Changed var type
	vector<MatrixXd> rawResults = run(agents, environments, maxEpisodes, maxEpisodeLengths, generator, numTrialsInExperiment, numTrialsTotal, experimentIDs, trialCounts);
	cout << "\tTrials completed." << endl;

	// Print the results to a file.
	cout << "Printing results to out/results.csv..." << endl;
#ifdef _MSC_VER	// Check if the compiler is a Microsoft compiler.
	//string filePath = "out/results.csv";	// If so, use this path
	string path = "out/results_";	// If so, use this path
#else
	//string filePath = "../out/results.csv";	// Otherwise, use this path
	string path = "../out/results_";	// Otherwise, use this path
#endif
	// TO-DO: instead of using numSamples to write the results to CSV, write all results (so that CSV contains all results) and use numSaples parameter only for plotting
    int trialID = 0;
	for (int experiment = 0; experiment < (int)numTrialsInExperiment.size(); experiment++)
	{
		string envName = envNames[trialID];
		string agentFullName = agents[trialID]->getName();
		string featureGenFullName = phis[trialID]->getName();
		int maxEps = maxEpisodes[trialID];

		// Print full results to csv file (rows: episodes, cols: trials); first col is episode number
        string fullFilePath = path + "full_" + to_string(numTrialsInExperiment[experiment]) + "_trials_" + envName + "_" + featureGenFullName + "_" + agentFullName + ".csv";
        ofstream outFullResults(fullFilePath);
		outFullResults << fixed << setprecision(5);
		for (int epCount = 0; epCount < maxEps; epCount++)
		{
			outFullResults << epCount << ",";
			for (int trialCount = 0; trialCount < numTrialsInExperiment[experiment]; trialCount++)
			{
				outFullResults << rawResults[experiment](trialCount, epCount) << ",";
			}
			outFullResults << endl;
		}

		// Print summary results (for plots)
		string summaryFilePath = path + "summary_" + to_string(numTrialsInExperiment[experiment]) + "_trials_" + envName + "_" + featureGenFullName + "_" + agentFullName + ".csv";
		ofstream outSummaryResults(summaryFilePath);
		outSummaryResults << "Episode,Average Discounted Return,Standard Error" << endl;
		for (int epCount = 0; epCount < maxEps; epCount++)
		{
			double meanResult = rawResults[experiment].col(epCount).mean();
			double standardError = sampleStandardError(rawResults[experiment].col(epCount));

			outSummaryResults << epCount << "," << meanResult << "," << standardError << endl;
		}

		trialID += numTrialsInExperiment[experiment];
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

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
// START V(-1) [ORIGINAL] - WORKS!
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

/*
#include "stdafx.hpp"

using namespace std;
using namespace Eigen;


// Run numTrials agent lifetimes on the provided environment. The entry in position (i,j) of the resulting matrix is the return from the j'th episode in the i'th trial.

MatrixXd run(vector<Agent*> agents, vector<Environment*> environments, int numTrials, int numEpisodes, 
	int maxEpisodeLength, mt19937_64& generator)
{
	// Ensure that agents and environments are of length numTrials
	if ((agents.size() != numTrials) || (environments.size() != numTrials))
		errorExit("Error in run(...). The number of agents/environments did not match numTrials.");

	// Create the object that we will return
	MatrixXd result(numTrials, numEpisodes);

	// Each thread has its own random number generator, since generators are not thread safe
	vector<mt19937_64> generators(numTrials);
	for (int i = 0; i < numTrials; i++)
		generators[i].seed(generator());	// See with a random sample from the generator passed as an argument to this function

	cout << "\tThere are " << numTrials << " trials to run. Printing a * when each is completed..." << endl;
	// Loop over trials
#pragma omp parallel for	// This line instructs the compiler to parallelize the following for-loop. This uses openmp.
	for (int trial = 0; trial < numTrials; trial++)
	{
		// Run an agent lifetime
		double gamma = environments[trial]->getGamma();

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
				for (int t = 0; t < maxEpisodeLength; t++) // After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
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
					if (environments[trial]->episodeOver(generators[trial]))
					{
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
				result(trial, epCount) = G;
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
				for (int t = 0; t < maxEpisodeLength; t++) // After maxEpisodeLength the episode doesn't "end", we just stop simulating it - so we don't do a terminal update
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
				result(trial, epCount) = G;
			}
		}
		// End of a trial - print a star
		cout.put('*');
		cout.flush();
	}
	cout << endl; // We just printed a bunch of *'s. Put a newline so anything that prints after this starts on a new line.
	return result;
}

void sandbox()
{
	cout << "Running other experiments/code." << endl;
	//sandboxJune16_2023();
	//sandboxJune16_2023();

	// Test softmax variants
	VectorXd x(3);
	x[0] = 0;
	x[1] = 1000.2;
	x[2] = -3000.7;
	VectorXd p;
	softmax(x, p);
	cout << "Using the fancy method:" << endl << setprecision(9) << p.transpose() << endl;

	softmaxDebug(x, p);
	cout << "Using the regular method:" << endl << setprecision(9) << p.transpose() << endl;

	cout << "Done running other experiments/code. Hit enter to continue." << endl;
	(void)getchar();
}

int main(int argc, char* argv[])
{
	// Comment out the line below if you don't want to run other random experiments first!
//	sandbox();

	// Default random number generator
	mt19937_64 generator;

	// Hyperparameters
	int numTrials = 10, numAlgs = 1, numRuns = numTrials * numAlgs;			// DONE! @TODO: Reverse variable names. numTrials is the number of times each algorithm/environmnet pair is run, numRuns is the total number of runs that will happen.
	int numSamples = 1; // How many samples (average) we use for the plot
	int iOrder = 3, dOrder = 3;
	double alphaAC = 0.0001, betaAC = 0.0001, lambdaAC = 0.8;
	double alphaSarsa = 0.0001, lambdaSarsa = 0.8, epsilonSarsa = 0.01;
	double alphaQ = 0.0001, lambdaQ = 0.8, epsilonQ = 0.01;
	double alphaExpectedSarsa = 0.0001, lambdaExpectedSarsa = 0.8, epsilonExpectedSarsa = 0.01;
	double alphaReinforce = 0.01;

	// Create the environment objects
	cout << "Creating environments..." << endl;
	vector<Environment*> environments(numRuns);
	for (int i = 0; i < numRuns; i++)
		environments[i] = new MountainCar();
	//environments[i] = new CartPole();
		//environments[i] = new Acrobot();
		//environments[i] = new Gridworld687();
	cout << "\tEnvironments created." << endl;

	// Get parameters of the environment
	int observationDimension = environments[0]->getObservationDimension(), numActions = environments[0]->getNumActions(),
		maxEpisodes = environments[0]->getRecommendedMaxEpisodes(), maxEpisodeLength = environments[0]->getRecommendedEpisodeLength();
	double gamma = environments[0]->getGamma();
	VectorXd observationLowerBound = environments[0]->getObservationLowerBound(),
		observationUpperBound = environments[0]->getObservationUpperBound();

	// Create agents. First, we need the FeatureGenerator objects - one for each!
	cout << "Creating feature generators..." << endl;
	vector<FeatureGenerator*> phis(numRuns);
	for (int i = 0; i < numRuns; i++)
		phis[i] = new FourierBasis(observationDimension, observationLowerBound, observationUpperBound, iOrder, dOrder);
	cout << "\tFeatures generators created." << endl;

	// Now, actually create the agents
	cout << "Creating agents..." << endl;
	vector<Agent*> agents(numRuns);
	for (int i = 0; i < numTrials; i++)
	{
		//agents[i] = new ActorCritic(observationDimension, numActions, alphaAC, betaAC, lambdaAC, gamma, phis[i]);
		//agents[i + numTrials] = new SarsaLambda(observationDimension, numActions, alphaSarsa, lambdaSarsa, epsilonSarsa, gamma, phis[i + numTrials]);
		//agents[i + 2 * numTrials] = new QLambda(observationDimension, numActions, alphaQ, lambdaQ, epsilonQ, gamma, phis[i + 2 * numTrials]);
		//agents[i + 3 * numTrials] = new ExpectedSarsaLambda(observationDimension, numActions, alphaQ, lambdaQ, epsilonQ, gamma, phis[i + 3 * numTrials]);


		// Assumes numTrials == numRuns (testing 1 algorithm)
        agents[i] = new ActorCritic(observationDimension, numActions, alphaAC, betaAC, lambdaAC, gamma, phis[i]);
//		agents[i] = new Reinforce(observationDimension, numActions, alphaReinforce, gamma, phis[i]);
	}
	cout << "\tAgents created." << endl;

	// Get names
	string environmentName = environments[0]->getName();

	// Actually run the trials - this function is threaded!
	cout << "Running trials..." << endl;
	MatrixXd rawResults = run(agents, environments, numRuns, maxEpisodes, maxEpisodeLength, generator);
	cout << "\tTrials completed." << endl;

	// Print the results to a file.
	cout << "Printing results to out/results.csv..." << endl;
#ifdef _MSC_VER	// Check if the compiler is a Microsoft compiler.
	//string filePath = "out/results.csv";	// If so, use this path
	string path = "out/results-" + to_string(numTrials) + " trials-";	// If so, use this path
#else
	//string filePath = "../out/results.csv";	// Otherwise, use this path
	string path = "../out/results";	// Otherwise, use this path
#endif

	for (int algCount = 0; algCount < numAlgs; algCount++)
	{
		// Get the name of the algorithm
		string algName = agents[algCount * numTrials]->getName();
		string filePath = path + environmentName + " with iO " + to_string(iOrder) + ", dO " + to_string(dOrder) + "-" + algName + ".csv";
		ofstream outResults(filePath);

		double meanResult = 0, meanStandardError = 0;

		outResults << "Episode,Average Discounted Return,Standard Error" << endl;
		for (int epCount = 0; epCount < maxEpisodes; epCount++)
		{
			MatrixXd algResult = rawResults.block(algCount * numTrials, 0, numTrials, maxEpisodes);
			meanResult += algResult.col(epCount).mean();
			meanStandardError += sampleStandardError(algResult.col(epCount));
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
// END V(-1) [ORIGINAL]
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

*/

