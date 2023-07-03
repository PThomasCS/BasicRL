#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

AlMountainCar::AlMountainCar()
{
    x = v = 0;
}

int AlMountainCar::getObservationDimension() const
{
    return 2;
}

int AlMountainCar::getNumActions() const
{
    return 3;
}

double AlMountainCar::getGamma() const
{
    return 1.0;
}

int AlMountainCar::getRecommendedEpisodeLength() const
{
    return 10000;
}

int AlMountainCar::getRecommendedMaxEpisodes() const
{
    return 1000;
}

string AlMountainCar::getName() const
{
    return "Mountain Car";
}

VectorXd AlMountainCar::getObservationLowerBound() const
{
    VectorXd lb(2);
    lb << -1.2, 0.5;
    return lb;
}

VectorXd AlMountainCar::getObservationUpperBound() const
{
    VectorXd ub(2);
    ub << 0.5, 0.7;
    return ub;
}

void AlMountainCar::newEpisode(mt19937_64& generator)
{
    // Restart with initial state (x, 0), where x is drawn uniformly at random from [-0.6, -0.4]
    uniform_real_distribution<> uniformInitialState(-0.6, -0.4);
    x = uniformInitialState(generator);
    v = 0;
}

void AlMountainCar::getObservation(mt19937_64& generator, VectorXd& buff) const
{
    buff = VectorXd::Zero(2);
    buff(0) = x;
    buff(1) = v;
}

double AlMountainCar::step(int action, mt19937_64& generator)
{
    // Shift action from 0, 1, 2 to -1, 0, 1
    int realAction = action - 1;

    // Handle the action
    if (realAction == -1 || realAction == 0 || realAction == 1)
    {
        v = v + 0.001 * realAction - 0.0025 * cos(3 * x);
        x = x + v;
    }

    else
    {
        assert(false);
        errorExit("Error in Mountain Car::step. Unrecognized action.");
    }

    // Make sure x stays in [-1.2, 0.5] and v is in [-0.7, 0.7].
    if (x < -1.2)
    {
        x = -1.2;
        v = 0;              // Simulate an inelastic collision when reaching the left bound
    }
    else if (x > 0.5)
    {
        x = 0.5;
        v = 0;              // Simulate an inelastic collision when reaching the right bound
    }

    if (v < -0.7)
        v = 0.7;
    else if (v > 0.7)
        v = 0.7;

    // Return a reward of -1 for each time step.
    return -1.0;
}
bool AlMountainCar::episodeOver(mt19937_64& generator) const
{
    return (x == 0.5);
}
