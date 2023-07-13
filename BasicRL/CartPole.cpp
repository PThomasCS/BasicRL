#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

CartPole::CartPole()
{
    x = v = theta = omega = 0;
}

int CartPole::getObservationDimension() const
{
    return 4;
}

int CartPole::getNumActions() const
{
    return 2;
}

double CartPole::getGamma() const
{
    return 1.0;
}

int CartPole::getRecommendedEpisodeLength() const
{
    return 1000; // 20 sec max, one time step is 0.02 sec
}

int CartPole::getRecommendedMaxEpisodes() const
{
    return 500;
}

string CartPole::getName() const
{
    return "Cart-Pole";
}

VectorXd CartPole::getObservationLowerBound() const
{
    VectorXd lb(4);
    lb << -3.0, -1.0*numeric_limits<float>::infinity(), -M_PI, -1.0*numeric_limits<float>::infinity();
    return lb;
}

VectorXd CartPole::getObservationUpperBound() const
{
    VectorXd ub(4);
    ub << 3.0, numeric_limits<float>::infinity(), M_PI, numeric_limits<float>::infinity();
    return ub;
}

void CartPole::newEpisode(mt19937_64& generator)
{
    x = v = theta = omega = 0;
}

void CartPole::getObservation(mt19937_64& generator, VectorXd& buff) const
{
    buff.resize(4); // After this line, the contents of buff(0) and buff(1) could be anything. We don't waste the time loading with zeros, because we are about to overwrite both entries
    buff(0) = x;
    buff(1) = v;
    buff(2) = theta;
    buff(3) = omega;
}

double CartPole::step(int action, mt19937_64& generator) {
    // Convert action idx to force applied to cart
    if (action < 0 || action > 1) {
        assert(false);
        errorExit("Error in CartPole::step. Unrecognized action.");
    }

    double F = -10.0;

    if (action == 1)
        F = -10.0;

    // x    : (x     , v   , theta,          omega   )
    // xDot : (xDot=v, vDot, thetaDot=omega, omegaDot)

    double omegaDot = (9.8 * sin(theta) + cos(theta) * ((-F - 0.05 * omega * omega * sin(theta)) / (1.1))) /
                      (0.5 * ((4.0 / 3.0) - ((0.1 * cos(theta) * cos(theta)) / (0.1))));
    double vDot = (F + 0.05) / (omega * omega * sin(theta) - omegaDot * cos(theta));

    // Update the state: x_(t+1) = x_t + 0.02*xDot_t

    x = max(-3.0, min(3.0, x + 0.02 * v));     // Keep cart's position in bounds [-3.0, 3.0]
    v = v + 0.02 * vDot;
    theta = theta + 0.02 * omega;
    omega = omega + 0.02 * omegaDot;

    // Keep pole's angle within [-pi, pi] range
    theta = fmod(theta, 2.0 * M_PI);
    if (theta <= -M_PI)
        theta += 2.0 * M_PI;
    else if (theta > M_PI)
        theta -= 2.0 * M_PI;

    // Return reward 1 for each time step (0.02 sec)
    return 1;
}

bool CartPole::episodeOver(mt19937_64& generator) const
{
    return (fabs(theta) >= M_PI / 15.0);       // fabs() for abs value
}

