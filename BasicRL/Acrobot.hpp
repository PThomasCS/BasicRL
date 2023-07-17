#pragma once	// Avoid recursive #include issues

#include "stdafx.hpp"

class Acrobot : public Environment {
public:
    ////////////////////////////////////////////////////////////////
    // Constructor
    ////////////////////////////////////////////////////////////////
    Acrobot();

    ////////////////////////////////////////////////////////////////
    // Functions for getting properties of the environment
    ////////////////////////////////////////////////////////////////
    virtual int getObservationDimension() const override;                // Get the dimension of the observation vector
    virtual int getNumActions() const override;                            // Get the number of actions
    virtual double getGamma() const override;                            // Get the discount factor
    virtual int
    getRecommendedEpisodeLength() const override;            // Get recommended episode length. Episodes might be terminated after this many time steps external to the environment (episodeOver will not necessarily return true).
    virtual int
    getRecommendedMaxEpisodes() const override;                // Get the recommended maximum number of episodes for an agent lifetime for this environment.
    virtual std::string getName() const override;                        // Get the name of the environment
    virtual Eigen::VectorXd
    getObservationLowerBound() const override;    // Get a lower bound on each observation feature
    virtual Eigen::VectorXd
    getObservationUpperBound() const override;    // Get a lower bound on each observation feature

    ////////////////////////////////////////////////////////////////
    // Functions for interacting with the environment
    ////////////////////////////////////////////////////////////////
    virtual void newEpisode(std::mt19937_64 &generator) override;                                        // new episode
    virtual void
    getObservation(std::mt19937_64 &generator, Eigen::VectorXd &buff) const override;        // get observation
    virtual double step(int action,
                        std::mt19937_64 &generator) override;                                // step from time (t) to time (t+1), where the agent selects action 'a', return the resulting reward
    virtual bool episodeOver(
            std::mt19937_64 &generator) const override;                                // query whether the episode is over (only call once per time step).

private:
    // State variables
    double theta1;      // Position of joint 1 ("shoulder")
    double theta2;      // Position of joint 2 ("elbow")
    double theta1Dot;   // Angular velocity of joint 1 ("shoulder")
    double theta2Dot;   // Angular velocity of 2 ("elbow")

     	// Standard parameters for the Acrobot
    const double m1 = 1.0;				// Mass of the first link
    const double m2 = 1.0;				// Mass of the second link
    const double l1 = 1.0;				// Length of the first link
    const double l2 = 1.0;				// Length of the second link
    const double lc1 = 0.5;
    const double lc2 = 0.5;
    const double i1 = 1.0;
    const double i2 = 1.0;
    const double g = 9.8;				// Acceleration due to gravity
    const double fmax = 1.0;		    // Maximum (and -minimum) force that can be applied
    const double dt = 0.05;				// Time step duration (CHANGE FROM BOOK)
    const double numSimSteps = 4;       // Number of steps to simulate, each of length dt
};