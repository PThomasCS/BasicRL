#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

AlGridworld687::AlGridworld687()
{
    x = y = 0;
}

int AlGridworld687::getObservationDimension() const
{
    return 24;	// A one-hot encoding of the state (ignoring the goal state)
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
    return "Alexandra Gridworld687";
}

VectorXd AlGridworld687::getObservationLowerBound() const
{
    return VectorXd::Zero(24);
}

VectorXd AlGridworld687::getObservationUpperBound() const
{
    return VectorXd::Ones(24);
}

void AlGridworld687::newEpisode(mt19937_64& generator)
{
    x = y = 0;
}

void AlGridworld687::getObservation(mt19937_64& generator, VectorXd& buff) const
{
    buff = VectorXd::Zero(24);
    buff(y*5 + x) = 1.0;
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
    return false;
}

double AlGridworld687::step(int action, mt19937_64& generator) {
    // Make sure the action is legal
    if ((action < 0) || (action > 3))
    {
        assert(false);
        errorExit("Error in AlGridworld687::step. Unrecognized action.");
    }

    // Handle the action
    int numActions = 4;
    int realAction;
    bool willHitObstacle;
    bool brokeDown = false;

    double randNum = uniform_real_distribution<double>(0.0, 1.0)(generator);

    // Transform intended action into real action
    if (randNum < 0.8)                                           // Move in the specified direction
        realAction = action;
    else if (randNum < 0.85)
        realAction = (action + 1) % numActions;                 // Veer Right of intended direction
    else if (randNum < 0.9)
        realAction = (action == 0 ? 3 : action - 1);            // Veer Left of intended direction
    else {
        realAction = action;
        brokeDown = true;                                       // Break down
    }

    willHitObstacle = hitObstacle(realAction);

    // Execute real action
    // TO-DO: handle obstacle states in less complicated way (?); make it work with custom obstacle states
    if ((!brokeDown) && (!willHitObstacle))
    {
        if (realAction == 0)   // Move UP
            y--;
        else if (realAction == 1)   // Move RIGHT
            x++;
        else if (realAction == 2)   // Move Down
            y++;
        else if (realAction == 3)   // Move LEFT  unless will enter an obstacle state
            x--;
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
    return 0.0;
 }

bool AlGridworld687::episodeOver(mt19937_64& generator) const
{
    return ((x == 4) && (y == 4));
}