#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

Sepsis::Sepsis(vector<vector<double>> transitionProbabilities, vector<vector<double>> rewards, vector<double> initialStateDistribution)
{
    // Use this for Mac
    //transitionProbabilities = readCSVToMatrix("/Users/alexandraburushkina/Desktop/Data/Code/BasicRL/BasicRL/icu-sepsis_params/tx_mat.csv");                      // Matrix with shape (numStates*numActions, numStates)
    //rewards = readCSVToMatrix("/Users/alexandraburushkina/Desktop/Data/Code/BasicRL/BasicRL/icu-sepsis_params/r_mat.csv");                                       // Matrix with shape (numStates*numActions, numStates)
    //initialStateDistribution = convertTo1D(readCSVToMatrix("/Users/alexandraburushkina/Desktop/Data/Code/BasicRL/BasicRL/icu-sepsis_params/d_0.csv"));           // Vector with shape (numStates)

    // This no longer works (?)
    //transitionProbabilities = readCSVToMatrix(R"(C:\Data\BasicRL\BasicRL\icu - sepsis_params\tx_mat.csv)");                                                         // Matrix with shape (numStates*numActions, numStates)
    //rewards = readCSVToMatrix(R"(C:\Data\BasicRL\BasicRL\icu - sepsis_params\r_mat.csv)");                                                                          // Matrix with shape (numStates*numActions, numStates)
    //initialStateDistribution = convertTo1D(readCSVToMatrix(R"(C:\Data\BasicRL\BasicRL\icu - sepsis_params\d_0.csv)"));                                              // Vector with shape (numStates)

    // Use this for WALL-E; works now
    //transitionProbabilities = readCSVToMatrix("tx_mat.csv");                                                         // Matrix with shape (numStates*numActions, numStates)
    //rewards = readCSVToMatrix("r_mat.csv");                                                                          // Matrix with shape (numStates*numActions, numStates)
    //initialStateDistribution = convertTo1D(readCSVToMatrix("d_0.csv"));                                              // Vector with shape (numStates)

    this->transitionProbabilities = transitionProbabilities;
    this->rewards = rewards;
    this->initialStateDistribution = initialStateDistribution;
}

int Sepsis::getObservationDimension() const
{
    return 747;
}

int Sepsis::getNumActions() const
{
    return 25; // Ask Kartik: what is the allowable actions file? Do we need to use that.
}

double Sepsis::getGamma() const
{
    return 1; // Try < 1 at some point
}

int Sepsis::getRecommendedEpisodeLength() const
{
    return 10000; // was 10000
}

int Sepsis::getRecommendedMaxEpisodes() const
{
    return 1000; // was 1000
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