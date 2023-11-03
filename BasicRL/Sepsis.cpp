#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

Sepsis::Sepsis()
{
    transitionProbabilities = readCSVToMatrix("/Users/alexandraburushkina/Desktop/Data/Code/BasicRL/BasicRL/icu-sepsis_params/tx_mat.csv");                      // Matrix with shape (numStates*numActions, numStates)
    rewards = readCSVToMatrix("/Users/alexandraburushkina/Desktop/Data/Code/BasicRL/BasicRL/icu-sepsis_params/r_mat.csv");                                       // Matrix with shape (numStates*numActions, numStates)
    initialStateDistribution = convertTo1D(readCSVToMatrix("/Users/alexandraburushkina/Desktop/Data/Code/BasicRL/BasicRL/icu-sepsis_params/d_0.csv"));    // Vector with shape (numStates)
}

int Sepsis::getObservationDimension() const
{
    return 747;	// A one-hot encoding of the state (ignoring the goal state) ???
}

int Sepsis::getNumActions() const
{
    return 25;
}

double Sepsis::getGamma() const
{
    return 1;
}

int Sepsis::getRecommendedEpisodeLength() const
{
    return 1000; //  was 10000
}

int Sepsis::getRecommendedMaxEpisodes() const
{
    return 3; // was 1000
}

string Sepsis::getName() const
{
    return "Sepsis";
}

VectorXd Sepsis::getObservationLowerBound() const
{
    return VectorXd::Zero(747);
}

VectorXd Sepsis::getObservationUpperBound() const
{
    return VectorXd::Ones(747);
}

void Sepsis::newEpisode(mt19937_64& generator)
{
    state = randp(initialStateDistribution, generator);
}

void Sepsis::getObservation(mt19937_64& generator, VectorXd& buff) const
{
    buff = VectorXd::Zero(747);
    buff(state) = 1.0;               
}

double Sepsis::step(int action, mt19937_64& generator) {
    // Make sure the action is legal
    if ((action < 0) || (action > 25))
    {
        assert(false);
        errorExit("Error in Sepsis::step. Unrecognized action.");
    }

    // Handle the action
    int sPrime = randp(transitionProbabilities[state*25 + action], generator);
    // Return a reward for entering sPrime:
    double reward = rewards[state*25 + action][sPrime];
    state = sPrime;
    return reward;
}

bool Sepsis::episodeOver(mt19937_64& generator) const
{
    return ((state > 743));
}