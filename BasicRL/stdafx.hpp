#pragma once	// Avoid recursive #include issues. Put this at the top of every header (.h and .hpp) file.

/*
* This file has all of our include statements in it (and definitions).
* Other files don't need to include everything - they can just include this!
*/

////////////////////////////////////////////////////////////////
// Definitions that modify subsequent includes
////////////////////////////////////////////////////////////////
#define _USE_MATH_DEFINES	// Have math.h include constants like M_PI

////////////////////////////////////////////////////////////////
// Standard includes
////////////////////////////////////////////////////////////////
#include <iostream>					// For console input/output via cout, cin, cerr, etc.
#include <random>					// For random numbers using generators like mt19937_64
#include <vector>					// For vector data structures - not for linear algebra
#include <string>					// For using strings as opposed to char* (the old C++ way).
#include <math.h>					// For some math terms like M_PI
#include <fstream>					// For reading from and printing to files, just like with cout/cin.

////////////////////////////////////////////////////////////////
// Additional libraries
////////////////////////////////////////////////////////////////
#include <Eigen/Dense>

////////////////////////////////////////////////////////////////
// General purpose files
////////////////////////////////////////////////////////////////
#include "common.hpp"				// Common general-purpose functions

////////////////////////////////////////////////////////////////
// Feature generators like the Fourier basis
////////////////////////////////////////////////////////////////
#include "FeatureGenerator.hpp"
#include "FourierBasis.hpp"

////////////////////////////////////////////////////////////////
// Environment files
////////////////////////////////////////////////////////////////
#include "Environment.hpp"			// Environment specification
#include "Gridworld.hpp"			// A simple gridworld for initial testing
#include "AlGridworld687.hpp"       // 5x5 Gridworld with obstacles and water state
#include "AlMountainCar.hpp"        // Mountain Car
#include "CartPole.hpp"             // CartPole (from 2019 Course Notes)

////////////////////////////////////////////////////////////////
// Agent files
////////////////////////////////////////////////////////////////
#include "Agent.hpp"				// Agent specification
#include "SarsaLambda.hpp"			// Linear Sarsa(lambda)
#include "AlQLambda.hpp"			// Q(lambda)
#include "AlActorCritic.hpp"        // Actor-Critic

////////////////////////////////////////////////////////////////
// Sandbox / Experiments not necessarily related to RL
////////////////////////////////////////////////////////////////
#include "June16_2023.hpp"