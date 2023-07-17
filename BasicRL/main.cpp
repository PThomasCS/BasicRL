#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

/*
* Run numTrials agent lifetimes on the provided environment. The entry in position (i,j) of the resulting matrix is the return from the j'th episode in the i'th trial.
*/
MatrixXd run(vector<Agent*> agents, vector<Environment*> environments, int numTrials, int numEpisodes, int maxEpisodeLength, mt19937_64& generator)
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

// TODO: I think we should delete the function below once the one above is set up properly. It is similar, but threaded!

// Takes as input an agent and an environment, runs the agent on the environment for one agent lifetime.
// Returns the vector of returns from each episode.
VectorXd runLifetime(Agent* agent, Environment* env, int numEpisodes, int maxEpisodeLength, 
	mt19937_64 & generator, VectorXd & totalTimeStepsAtEpisodeEndsBuff)	// Buff means "buffer" - we are going to return values in this object.
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
	sandboxJune16_2023();
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
	int numTrials = 30, iOrder = 3, dOrder = 3;
//    int numRuns = 2, iOrder = 3, dOrder = 3;
	double alphaAC = 0.0001, betaAC = 0.0001, lambdaAC = 0.8;
	double alphaSarsa = 0.0001, lambdaSarsa = 0.8, epsilonSarsa = 0.1;
	double alphaQ = 0.0001, lambdaQ = 0.8, epsilonQ = 0.1;
//    double alpha = 0.001, beta = 0.001, lambda = 0.8, epsilon = 0.05;
//    double alpha = 0.001, lambda = 0.8, epsilon = 0.05;

//    int numAlgorithms = 3;
//    int numTrials = numRuns * numAlgorithms;
  // comment
	// Create the environment objects
	cout << "Creating environments..." << endl;
	vector<Environment*> environments(numTrials);
	for (int i = 0; i < numTrials; i++)
//		environments[i] = new MountainCar();
      environments[i] = new CartPole();
//        environments[i] = new Acrobot();
	cout << "\tEnvironments created." << endl;

	// Get parameters of the environment
	int observationDimension = environments[0]->getObservationDimension(), numActions = environments[0]->getNumActions(),
		maxEpisodes = environments[0]->getRecommendedMaxEpisodes(), maxEpisodeLength = environments[0]->getRecommendedEpisodeLength();
	double gamma = environments[0]->getGamma();
	VectorXd observationLowerBound = environments[0]->getObservationLowerBound(),
		observationUpperBound = environments[0]->getObservationUpperBound();

	// Create agents. First, we need the FeatureGenerator objects - one for each!
	cout << "Creating feature generators..." << endl;
	vector<FeatureGenerator*> phis(numTrials);
	for (int i = 0; i < numTrials; i++)
		phis[i] = new FourierBasis(observationDimension, observationLowerBound, observationUpperBound, iOrder, dOrder);
	cout << "\tFeatures generators created." << endl;

	// Now, actually create the agents
	cout << "Creating agents..." << endl;
	vector<Agent*> agents(numTrials);
//    for (int i = 0; i < numTrials; i++)
//        agents[i] = new ActorCritic(observationDimension, numActions, alpha, beta, lambda, gamma, phis[i]);

	if (numTrials != 30) errorExit("Running code that is hard-coded for numTrials=30 without numTrials=30.");

    for (int i = 0; i < 10; i++)
        agents[i] = new ActorCritic(observationDimension, numActions, alphaAC, betaAC, lambdaAC, gamma, phis[i]);
    for (int i = 10; i < 20; i++)
        agents[i] = new SarsaLambda(observationDimension, numActions, alphaSarsa, lambdaSarsa, epsilonSarsa, gamma, phis[i]);
    for (int i = 20; i < 30; i++)
        agents[i] = new QLambda(observationDimension, numActions, alphaQ, lambdaQ, epsilonQ, gamma, phis[i]);

//	for (int i = 0; i < numTrials; i++)
//		agents[i] = new ActorCritic(observationDimension, numActions, alpha, beta, lambda, gamma, phis[i]); // The &phi means "the memory location of phi". Notice the constructor takes a pointer FeatureGenerator*.
////        agents[i] = new SarsaLambda(observationDimension, numActions, alpha, lambda, epsilon, gamma, phis[i]);
	cout << "\tAgents created." << endl;

	// Actually run the trials - this function is threaded!
	cout << "Running trials..." << endl;
	MatrixXd rawResults = run(agents, environments, numTrials, maxEpisodes, maxEpisodeLength, generator);
	cout << "\tTrials completed." << endl;
	
	// Print the results to a file.
	cout << "Printing results to out/results.csv..." << endl;
#ifdef _MSC_VER	// Check if the compiler is a Microsoft compiler.
	string filePath = "out/results.csv";	// If so, use this path
#else
	string filePath = "../out/results.csv";	// Otherwise, use this path
#endif
	ofstream outResults(filePath);
	outResults << "Episode,Average Discounted Return,Standard Error" << endl;
	for (int epCount = 0; epCount < maxEpisodes; epCount++)
		outResults << epCount << "," << sampleMean(rawResults.col(epCount)) << "," << sampleStandardError(rawResults.col(epCount)) << endl;	// The functions 'sampleMean' and 'sampleStandardError' are defined in common.hpp
	outResults.close();
	cout << "\tResults printed." << endl;

	// Clean up memory. Everything that we called "new" for, we need to call "delete" for.
	for (int i = 0; i < numTrials; i++)
	{
		delete environments[i];
		delete phis[i];
		delete agents[i];
	}

//    system("learning_curves.py");

	// Print message indicating that the program has finished
	cout << "Done. Press enter to exit." << endl;
	return 0; // No error.
}