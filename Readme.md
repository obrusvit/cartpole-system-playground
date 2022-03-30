# Cartpole system playground

This repo contains a simulation of a cartpole system with various control engineering stuff. It's meant to refresh a few concepts and test capabilities of Julia.

## Simple simulation from initial conditions

Function `main_just_simulate()` to simulate cartpole system from initial conditions (stored in `CartPoleState`) with given parameters (stored in `CartPoleParams` struct). Examples:


Unforced:

![Simple simulation without force - gif](output/just_sim_1.gif)
![Simple simulation without force - plot](output/just_sim_1.png)


With some force applied and shorter pole:

![Simple simulation with force - gif](output/just_sim_2.gif)
![Simple simulation with force - plot](output/just_sim_2.png)

## Estimations

Estimation of all 4 states from a single measurement of position `x`.
![State estimation with EKF - plot](output/estim_EKF_small.png)

## LQR control

Example of recovery from bad initial position and subsequent push at time 10s.

![LQR control - gif](output/lqr_1.gif)
![LQR control - plot](output/lqr_1.png)

## Swing-up maneuver

### Using nonlinear optimization

![Swing up maneuver using optimization - gif](output/swingup_optim.gif)
![Swing up maneuver using optimization - plot](output/swingup_optim.png)

### Using reinforcement learning
