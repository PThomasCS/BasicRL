#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

Gridworld::Gridworld()
{
	size = 4;			// Default to a size of 4
	x = y = 0;
}

Gridworld::Gridworld(int size)
{
	this->size = size;
	x = y = 0;
}

int Gridworld::getObservationDimension() const
{
	return size*size-1;	// A one-hot encoding of the state (ignoring the goal state)
}

int Gridworld::getNumActions() const
{
	return 4;
}

double Gridworld::getGamma() const
{
	return 1.0;
}

int Gridworld::getRecommendedEpisodeLength() const
{
	return 10000;
}

int Gridworld::getRecommendedMaxEpisodes() const
{
	return 1000;
}

string Gridworld::getName() const
{
	return "Gridworld (" + to_string(size) + 'x' + to_string(size) + ")";
}

VectorXd Gridworld::getObservationLowerBound() const
{
	return VectorXd::Zero(size * size - 1);
}

VectorXd Gridworld::getObservationUpperBound() const
{
	return VectorXd::Ones(size * size - 1);
}

void Gridworld::newEpisode(mt19937_64& generator)
{
	x = y = 0;
}

void Gridworld::getObservation(mt19937_64& generator, VectorXd& buff) const
{
	buff = VectorXd::Zero(size * size - 1);
	buff(y * size + x) = 1.0;
}

double Gridworld::step(int action, mt19937_64& generator)
{
	// Handle the action
	if (action == 0)
		x++;
	else if (action == 1)
		x--;
	else if (action == 2)
		y++;
	else if (action == 3)
		y--;
	else
	{
		assert(false);
		errorExit("Error in Gridworld::step. Unrecognized action.");
	}

	// Make sure the agent (x,y) position is in-bounds.
	x = min(size - 1, max(0, x));
	y = min(size - 1, max(0, y));

	// Return a reward of -1 for each time step.
	return -1.0;
}
bool Gridworld::episodeOver(mt19937_64& generator) const
{
	return ((x == size - 1) && (y == size - 1));
}