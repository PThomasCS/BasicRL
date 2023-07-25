#include "stdafx.hpp"

using namespace std;
using namespace Eigen;

Acrobot::Acrobot()
{
    theta1 = theta2 = theta1Dot = theta2Dot = 0;
}

int Acrobot::getObservationDimension() const
{
    return 4;
}

int Acrobot::getNumActions() const
{
    return 3;
}

double Acrobot::getGamma() const
{
    return 1.0;
}

int Acrobot::getRecommendedEpisodeLength() const
{
    return 10000;
}

int Acrobot::getRecommendedMaxEpisodes() const
{
    return 1000;
}

string Acrobot::getName() const
{
    return "Acrobot";
}

VectorXd Acrobot::getObservationLowerBound() const
{
    // (x, v, theta, omega)
    VectorXd lb(4);
    lb << -M_PI, -M_PI, -4.0 * M_PI, -9.0 * M_PI;

    return lb;
}

VectorXd Acrobot::getObservationUpperBound() const
{
    // (x, v, theta, omega)
    VectorXd ub(4);
    ub << M_PI, M_PI, 4.0 * M_PI, -9.0 * M_PI;
    return ub;
}

void Acrobot::newEpisode(mt19937_64& generator)
{
    theta1 = theta2 = theta1Dot = theta2Dot = 0;
}

void Acrobot::getObservation(mt19937_64& generator, VectorXd& buff) const
{
    buff.resize(4); // After this line, the contents of buff(0) and buff(1) could be anything. We don't waste the time loading with zeros, because we are about to overwrite both entries
    buff(0) = theta1;
    buff(1) = theta2;
    buff(2) = theta1Dot;
    buff(3) = theta2Dot;
}

double Acrobot::step(int action, mt19937_64& generator)
{
    // Convert action idx to force applied to cart
    if (action < 0 || action > 2)
    {
        assert(false);
        errorExit("Error in Acrobot::step. Unrecognized action.");
    }

    // Convert action idx to torque applied to joint 2
    double torque = action - 1.0;

    if (rungeKutta)
    {
        double h = dt / integSteps;

        VectorXd y(4), k1(4), k2(4), k3(4), k4(4);
        y[0] = theta1;
        y[1] = theta2;
        y[2] = theta1Dot;
        y[3] = theta2Dot;

        for (int i = 0; i < integSteps; i++)
        {   
            // k1
            k1 = f(y, torque);

            // k2
            k2 = f(k1, torque);
            k2 = y + (h / 2) * k1;

            // k3
            k3 = f(k2, torque);
            k3 = y + (h / 2) * k2;

            // k4
            k4 = f(k3, torque);
            k3 = y + h * k2;
        
        }
    }
    else
    {
        // x    : (theta1     , theta2     , theta1Dot,       theta2Dot)
        // xDot : (theta1Dot  , theta2Dot  , theta1DotDot,    theta2DotDot)

        // Calculate theta1DotDot and theta2DotDot
        double d1, d2, phi1, phi2, theta1DotDot, theta2DotDot;

        for (int count = 0; count < numSimSteps; count++)
        {
            d1 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * cos(theta2)) + i1 + i2;
            d2 = m2 * (lc2 * lc2 + l1 * lc2 * cos(theta2)) + i2;
            phi2 = (m2 * lc2 * g * cos(theta1 + theta2 - M_PI / 2.0));
            phi1 = (-m2 * l1 * lc2 * theta2Dot * theta2Dot * sin(theta2) - 2 * m2 * l1 * lc2 * theta2Dot * theta1Dot * sin(theta2) + (m1 * lc1 + m2 * l1) * g * cos(theta1 - M_PI / 2.0) + phi2);
            theta2DotDot = ((1.0 / (m2 * lc2 * lc2 + i2 - (d2 * d2) / d1)) * (torque + (d2 / d1) * phi1 - m2 * l1 * lc2 * theta1Dot * theta1Dot * sin(theta2) - phi2));
            theta1DotDot = ((-1.0 / d1) * (d2 * theta2DotDot + phi1));

            // Update the state: x_(t+1) = x_t + dt*xDot_t

            theta1 = theta1 + dt * theta1Dot;
            theta2 = theta2 + dt * theta2Dot;
            theta1Dot = max(-4.0 * M_PI, min(4.0 * M_PI, theta1Dot + dt * theta1DotDot)); // Keep theta1Dot in bounds [-4.0 / M_PI, 4.0 / M_PI]
            theta2Dot = max(-9.0 * M_PI, min(9.0 * M_PI, theta2Dot + dt * theta2DotDot)); // Keep theta2Dot in bounds [-9.0 / M_PI, 9.0 / M_PI]
        }

        // Enforce joint angle constraints
        theta1 = wrapPosNegPI(theta1);
        theta2 = wrapPosNegPI(theta2);

        // Return reward 1 for each time step (dt)
        return -1.0;
     }
}

bool Acrobot::episodeOver(mt19937_64& generator) const
{
    double elbowY = -l1 * cos(theta1);
    double handY = elbowY - l2 * cos(theta1 + theta2);
    return handY > l1;
}

VectorXd Acrobot::f(VectorXd buff, double torque)
{
    // x    : (theta1     , theta2     , theta1Dot,       theta2Dot)
    // xDot : (theta1Dot  , theta2Dot  , theta1DotDot,    theta2DotDot)

    double curTheta1 = buff[0];
    double curTheta2 = buff[1];
    double curTheta1Dot = buff[2];
    double curTheta2Dot = buff[3];

    // Calculate theta1DotDot and theta2DotDot
    double d1, d2, phi1, phi2, curTheta1DotDot, curTheta2DotDot;

    d1 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * cos(curTheta2)) + i1 + i2;
    d2 = m2 * (lc2 * lc2 + l1 * lc2 * cos(curTheta2)) + i2;
    phi2 = (m2 * lc2 * g * cos(curTheta1 + curTheta2 - M_PI / 2.0));
    phi1 = (-m2 * l1 * lc2 * curTheta2Dot * curTheta2Dot * sin(theta2) - 2 * m2 * l1 * lc2 * curTheta2Dot * curTheta1Dot * sin(curTheta2) + (m1 * lc1 + m2 * l1) * g * cos(curTheta1 - M_PI / 2.0) + phi2);
    curTheta2DotDot = ((1.0 / (m2 * lc2 * lc2 + i2 - (d2 * d2) / d1)) * (torque + (d2 / d1) * phi1 - m2 * l1 * lc2 * curTheta1Dot * curTheta1Dot * sin(curTheta2) - phi2));
    curTheta1DotDot = ((-1.0 / d1) * (d2 * curTheta2DotDot + phi1));

    VectorXd buffDot(4);

    buffDot[0] = curTheta1Dot;
    buffDot[1] = curTheta2Dot;
    buffDot[3] = curTheta1DotDot;
    buffDot[4] = curTheta2DotDot;

    return buffDot;
}