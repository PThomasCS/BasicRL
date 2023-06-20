#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

// TO-DO: add custom size, water, obstacle, and goal states.

AlGridworld687::AlGridworld687()
{
    size = 5;			// Default to a size of 5
    x = y = 0;
}

int AlGridworld687::getObservationDimension() const
{
    return size*size-1;	// A one-hot encoding of the state (ignoring the goal state)
}

int AlGridworld687::getNumActions() const
{
    return 4;
}

double AlGridworld687::getGamma() const
{
    return 0.9;
}

int AlGridworld687::getRecommendedEpisodeLength() const
{
    return 10000;
}

int AlGridworld687::getRecommendedMaxEpisodes() const
{
    return 1000;
}

string AlGridworld687::getName() const
{
    return "AlGridworld687";
}

VectorXd AlGridworld687::getObservationLowerBound() const
{
    return VectorXd::Zero(size * size - 1);
}

VectorXd AlGridworld687::getObservationUpperBound() const
{
    return VectorXd::Ones(size * size - 1);
}

void AlGridworld687::newEpisode(mt19937_64& generator)
{
    x = y = 0;
}

void AlGridworld687::getObservation(mt19937_64& generator, VectorXd& buff) const
{
    buff = VectorXd::Zero(size * size - 1);
    buff(y * size + x) = 1.0;
}

// Checks if the agent will end up in an obstacle state
bool AlGridworld687::hitObstacle(int realAction) const
{
    if (realAction == 0 && ((x == 2) && (y == 4)))
        return true;
    else if (realAction == 1 && ((x == 1) && ((y == 2) || (y == 3))))
        return true;
    else if (realAction == 2 && ((x == 2) && (y == 1)))
        return true;
    else if (realAction == 3 && ((x == 3) && ((y == 2) || (y == 3))))
        return true;
    else
    {
        return false;
    }
}

double AlGridworld687::step(int action, mt19937_64& generator) {
    // Handle the action
    int numActions = 4;
    int realAction;
    bool willHitObstacle = false;
    bool brokeDown = false;

    uniform_real_distribution<> uniformActionResultDistribution(0.0, 1.0);
    double result = uniformActionResultDistribution(generator);

    // Transform intended action into real action
    if (result < 0.8)                                           // Move in the specified direction
        realAction = action;
    else if (result < 0.85)
        realAction = (action + 1) % numActions;                 // Veer Right of intended direction
    else if (result < 0.9)
        realAction = (action - 1) % numActions;                 // Veer Left of intended direction
    else {
        realAction = action;
        brokeDown = true;                                       // Break down
    }

    willHitObstacle = hitObstacle(realAction);

    // Execute real action
    // TO-DO: handle obstacle states in less complicated way (?); make it work with custom obstacle states
    if (brokeDown)
    {
        // Do nothing
    }
    else if ((realAction == 0) && (willHitObstacle == false))   // Move UP unless will enter an obstacle state
        y--;
    else if ((realAction == 1) && (willHitObstacle == false))   // Move RIGHT unless will enter an obstacle state
        x++;
    else if ((realAction == 2) && (willHitObstacle == false))   // Move Down unless will enter an obstacle state
        y++;
    else if ((realAction == 3) && (willHitObstacle == false))   // Move LEFT  unless will enter an obstacle state
        x--;
    else {
        // TO-DO: write another error message
//        assert(false);
//        errorExit("Error in AlGridworld687::step. Unrecognized action.");
    }

    // TO-DO: try using the same logic as for obstacle states (don't move if you'll hit the wall)
    // Make sure the agent (x,y) position is in-bounds.
    x = min(4, max(0, x));
    y = min(4, max(0, y));

    // Return a reward for entering sPrime.
    if ((x == 2) && (y == 4)) // Water state
        return -10.0;
    else if ((x == 4) && (y == 4)) // Goal state
        return 10.0;
    else
    {
        return 0.0;
    }
 }

bool AlGridworld687::episodeOver(mt19937_64& generator) const
{
    return ((x == 4) && (y == 4));
}