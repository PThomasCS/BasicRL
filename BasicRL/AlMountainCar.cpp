#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

// Random initial state (uniformly at random (x, v) withing range) or a default deterministic initial state (-0.5, 0)
AlMountainCar::AlMountainCar(bool randomStart)
{
    x = v = 0;
    this->randomStart = randomStart;
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
    return 20000;
}

int AlMountainCar::getRecommendedMaxEpisodes() const
{
    return 5000;
}

string AlMountainCar::getName() const
{
    return "Alexandra Mountain Car";
}

VectorXd AlMountainCar::getObservationLowerBound() const
{
    VectorXd lb(2);
    lb << -1.2, -0.07;
    return lb;
}

VectorXd AlMountainCar::getObservationUpperBound() const
{
    VectorXd ub(2);
    ub << 0.5, 0.07;
    return ub;
}

void AlMountainCar::newEpisode(mt19937_64& generator)
{
    if (randomStart) // Restart with initial state (x, v), where x and v are drawn uniformly at random from the range
    {
        x = uniform_real_distribution<double>(-1.2, 0.5)(generator);
        // The line above is equivalent to the two lines below.
        //uniform_real_distribution<double> uniformX(-1.2, 0.5);
        //x = uniformX(generator);
        v = uniform_real_distribution<double>(-0.07, 0.07)(generator);
    }
    else
    {
        x = -0.5;
        v = 0;
    }
}

void AlMountainCar::getObservation(mt19937_64& generator, VectorXd& buff) const
{
    buff.resize(2); // After this line, the contents of buff(0) and buff(1) could be anything. We don't waste the time loading with zeros, because we are about to overwrite both entries
    buff(0) = x;
    buff(1) = v;
}

double AlMountainCar::step(int action, mt19937_64& generator)
{
    if (action < 0 || action > 2)
    {
        assert(false);
        errorExit("Error in AlMountainCar::step. Unrecognized action.");
    }

    // Shift action from 0, 1, 2 to -1, 0, 1
    double transition = double(action) - 1.0;

    // Handle the action
     v = v + 0.001 * transition - 0.0025 * cos(3.0 * x);
     x = x + v;

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

    if (v < -0.07)
        v = -0.07;
    else if (v > 0.07)
        v = 0.07;

    // Return a reward of -1 for each time step.
    return -1.0;
}

bool AlMountainCar::episodeOver(mt19937_64& generator) const
{
    return (x == 0.5);
}
