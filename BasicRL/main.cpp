#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

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

int main(int argc, char* argv[])
{
	// Comment out the line below if you don't want to run other random experiments first!
//	sandbox();
	
	// Set hyperparameters and RNG
    double  alpha = 0.001, beta = 0.001, lambda = 0.9, epsilon = 0.0, sigma = 0.5; // mc
//    double  alpha = 0.001, lambda = 0.9, epsilon = 0.0;
//	double  alpha = 0.01, beta = 0.01, lambda = 0.8, epsilon = 0.05, sigma = 0.1; // Qlambda, Actor-Critic, gr687
//	double  alpha = 0.1, beta = 0.1, lambda = 0.8, epsilon = 0.05, sigma = 0.1;

	int iOrder = 5, dOrder = 5; // mc 5, 5
	mt19937_64 generator;	// If you don't seed it, it has some fixed seed that is the same every time.

	// Create the environment
//	Gridworld env(4);
//  AlGridworld687 env;
//  AlMountainCar env;
    AlMountainCar env(false);

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
//    AlQLambda agent(observationDimension, numActions, alpha, lambda, epsilon, gamma, &phi); // The &phi means "the memory location of phi". Notice the constructor takes a pointer FeatureGenerator*.
    AlActorCritic agent(observationDimension, numActions, alpha, beta, lambda, gamma, sigma, &phi); // The &phi means "the memory location of phi". Notice the constructor takes a pointer FeatureGenerator*.

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
    ofstream outReturns("returns.txt");
	outReturns << returns << endl;
	outReturns.close();

    // Print message indicating that the program has finished
	cout << "Done. Press enter to exit." << endl;
	return 0; // No error.
}