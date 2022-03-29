"

 author: Obrusnik Vit
"

using DifferentialEquations
using Random, Distributions
using UnPack
using LinearAlgebra
using ControlSystems
using Plots
using LaTeXStrings
pyplot()

# import to prevent name clashes, e.g. @JuMP.variable vs. Symbolics.variable
import Symbolics
import JuMP

include("src/diffeq_simulation.jl")
include("src/postprocess_visualize.jl")
include("src/swingup_optim.jl")
include("src/swingup_rl.jl")
include("src/forcing_functions.jl")


function get_meas_noise()
    R_true = 0.1
    return rand(Normal(0.0, R_true))
end

function get_state_noise()
    means = [0.0; 0.0; 0.0; 0.0]
    Q_true = [0.001 0.0   0.0   0.0; 
              0.0   0.001 0.0   0.0;
              0.0   0.0   0.001 0.0;
              0.0   0.0   0.0   0.001]
    dist = MvNormal(means, Q_true)
    return rand(dist) 
end


function main_just_simulate()
    # CartPole structure creation
    # params = CartPoleParams();
    params  = CartPoleParams(
            1.0, 0.5,     # w, h
            1.0, 0.3,   # mt, mp
            1.0,        # L
            0.2, 0.2  # bt, bp
        )

    init = CartPoleState(0.0, 0.0, pi-0.3, 0.0);
    sys = CartPole(params, init);

    # Time
    tspan = (0.0, 20.0);
    Ts = 0.001;
    t_lin = range(tspan[begin], tspan[end], step=Ts);

    # # Forcing function
    f(x, t) = f_step(x, t; t_step_begin=4, t_step_end=6, weight=0.3);
    # f(x, t) = f0(x,t);

    # # ODE problem of nonlinear CartPole System
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f];
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇];
    prob = ODEProblem(cartPoleSystem, x0, tspan, p);
    saved_values = SavedValues(Float64, Vector{Float64});
    function cb_save(u,t,p) 
        f = p[end]
        return vcat(u, f(u, t));
    end
    cb = SavingCallback((u, t, integrator) -> cb_save(u, t, p), saved_values; saveat=t_lin);
    sol = solve(prob, callback=cb);

    # Plot
    plot_sol_force(t_lin, sol, saved_values; dest_name="just_sim.png")

    make_gif(sol, sys)
end


function main_estimator_cb()
    # CartPole structure creation
    params = CartPoleParams();
    init = CartPoleState(0.0, 0.0, pi - 0.1, 0.0);
    sys = CartPole(params, init);

    # Time
    tspan = (0.0, 20.0);
    Ts = 0.01;
    t_lin = range(tspan[begin], tspan[end], step=Ts);

    # LTI at lower pos
    sys_LTI = cartpole_LTI_sys(sys, CartPoleState(0, 0, 0, 0));
    sysd_LTI = c2d(sys_LTI, Ts, :zoh);

    # # Forcing function
    forcing_fun(x, t) = f_step(x, t; t_step_begin=4, t_step_end=6);

    # Callback for observer
    sys_obs = cartpole_observer(sys_LTI);
    sysd_obs = c2d(sys_obs, Ts, :zoh);
    u_obs = [0.0; 0.0; 0.1; 0.0];
    function cb_observer(u, t, p)
        noise = get_meas_noise()
        u[1] += noise
        f = p[end]
        # u[1] or sys_LTI_lower.C*u
        u_obs = sysd_obs.A * [u_obs[1]; u_obs[2]; u_obs[3]; u_obs[4]] + sysd_obs.B * [f(u, t); sys_LTI.C * u];
        return vcat(u_obs, u, f(u, t));
    end

    # Proper LKF
    P = 0.001   * I(4);
    Q = 0.00001 * I(4);
    R = 0.1 * I(1);

    A = sysd_LTI.A;
    B = sysd_LTI.B;
    C = sysd_LTI.C; 
    x = [0.0; 0.0; pi - 0.1; 0.0];

    function cb_LKF(u, t, p)
        noise = get_meas_noise()
        u[1] += noise  # measurement x is noisy
        f = p[end]

        # Meas update / Correction step / Data step
        y = C * u
        Lₖ = (P * C') / (C * P * C' + R)
        xₖ = x + Lₖ * (y - C * x);
        # Pₖ = P - Lₖ*(C*P*C')*Lₖ'
        Pₖ = P - Lₖ * C * P

        # Time update / Prediction step / Time step
        x = A * xₖ + B * f(u, t)
        P = A * Pₖ * A' + Q

        x_estim = vec(x);
        return vcat(x_estim, u, f(u, t));
    end

    function cb_LKF_2(u, t, p)
        noise = get_meas_noise()
        u[1] += noise
        f = p[end]

        # Time update / Prediction step / Time step
        xₖ = A * x + B * f(u, t)
        Pₖ = A * P * A' + Q

        # Meas update / Correction step / Data step
        y = C * u
        Lₖ = (Pₖ * C') / (C * Pₖ * C' + R)
        x = xₖ + Lₖ * (y - C * xₖ);
        P = Pₖ - Lₖ * (C * Pₖ * C') * Lₖ'

        x_estim = vec(x);
        return vcat(x_estim, u, f(u, t));
    end

    function cb_EKF(u, t, p)
        noise = get_meas_noise()
        u[1] += noise  # measurement x is noisy
        f = p[end]

        # Meas update / Correction step / Data step
        y = C * u
        Lₖ = (P * C') / (C * P * C' + R)
        xₖ = x + Lₖ * (y - C * x);
        Pₖ = P - Lₖ * (C * P * C') * Lₖ'
        # Pₖ = P - Lₖ*C*P

        # Time update / Prediction step / Time step
        sys.state = CartPoleState(xₖ[1], xₖ[2], xₖ[3], xₖ[4])
        sysₖ = cartpole_LTI_sys(sys, f(u, t))
        sysdₖ = c2d(sysₖ, Ts, :fwdeuler)
        Aₖ, Bₖ = sysdₖ.A, sysdₖ.B
        # Version 1: handrolled forward euler approximation
        # x = cartPoleDiscApprox(sys, f(u,t), 0, Ts)
        # Version 2: use DifferentialEquations to simulate one step
        x_sol = DifferentialEquations.solve(ODEProblem(cartPoleSystem, xₖ, (t, t + Ts), p))
        x = x_sol.u[end]

        P = Aₖ * Pₖ * Aₖ' + Q

        x_estim = vec(x);
        return vcat(x_estim, u, f(u, t));
    end

    # # ODE problem of nonlinear CartPole System
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, forcing_fun];
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇];
    prob = ODEProblem(cartPoleSystem, x0, tspan, p);
    saved_values = SavedValues(Float64, Vector{Float64});
    cb = SavingCallback((u, t, integrator) -> cb_EKF(u, t, p), saved_values; saveat=t_lin);
    sol = solve(prob, callback=cb);

    # Plot
    plot_sol_est_force(t_lin, sol, saved_values)
end

function main_LQR(gif::Bool)
    # CartPole structure creation
    params = CartPoleParams()
    init = CartPoleState(0.0, 0.0, pi - 0.8, 0.0)
    sys = CartPole(params, init)

    # time
    tspan = (0.0, 20.0)
    Ts = 0.01
    t_lin = range(tspan[begin], tspan[end], step=Ts)

    # LTI at upper pos
    sys_LTI_upper = cartpole_LTI_sys(sys, CartPoleState(0, 0, pi, 0)) 

    # # Forcing function
    Q = [0.1 0    0   0; 
         0   0.01 0   0;
         0   0    100   0;
         0   0    0   0.1]
    L_gain = lqr(sys_LTI_upper, Q, 0.1 * I)
    f(x, t) = forcing_1(x, t, L_gain)
    

    # # ODE problem of nonlinear CartPole System
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f]
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇]
    prob = ODEProblem(cartPoleSystem, x0, tspan, p)
    sol = solve(prob)

    # Plot
    plot_sol(t_lin, sol)
    if gif
        make_gif(sol, sys)
    end
end


function main_swingup_optim()
    # CartPole structure creation
    params = CartPoleParams(
            1.0, 0.5,     # w, h
            1.0, 0.1,   # mt, mp
            2.5,        # L
            0.5, 0.3  # bt, bp
        )
    init = CartPoleState(0.0)
    final = CartPoleState(pi)
    sys = CartPole(params, init)
    sys_LTI_upper = cartpole_LTI_sys(sys, final)

    T_N = 4.0  # final time of the maneuver
    N = 1001     # number of collocation points
    _, u = swingupControlTrajectory(T_N, N, params, init, final)

    gif_opt_sol = false
    if gif_opt_sol make_gif(collect(range(0.0, T_N, length=N)), x, sys) end

    f(x, t) = if t >= T_N || abs(x[3] - pi) <= 0.3
        force_LQR(x, sys_LTI_upper)
    else 
        force_swingup_optim(u, t, T_N)
    end

    # # ODE problem of nonlinear CartPole System
    tspan = (0, 3 * T_N)
    Ts = 0.001;
    t_lin = range(tspan[begin], tspan[end], step=Ts);
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f]
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇]
    prob = ODEProblem(cartPoleSystem, x0, tspan, p)
    saved_values = SavedValues(Float64, Vector{Float64});
    function cb_save(u,t,p) 
        f = p[end]
        return vcat(u, f(u, t));
    end
    cb = SavingCallback((u, t, integrator) -> cb_save(u, t, p), saved_values; saveat=t_lin);
    sol = solve(prob, callback=cb);

    # Plot
    plot_sol_force(t_lin, sol, saved_values; dest_name="swingup_optim.png")

    make_gif(sol, sys)
end

function main_swingup_rl(; nn_params=nothing)
    # CartPole structure creation
    params = CartPoleParams(
            1.0, 0.5,     # w, h
            1.0, 0.1,   # mt, mp
            1.0,        # L
            0.5, 0.3  # bt, bp
        )
    init = CartPoleState(0.0)
    final = CartPoleState(pi)
    sys = CartPole(params, init)
    sys_LTI_upper = cartpole_LTI_sys(sys, final)

    T_N = 1.0  # final time of the maneuver
    N = 401
    if nn_params == nothing
        nn_params = trainCartPoleController(T_N, N, params, init, final)
        println("training finished")
    end

    f(x, t) = if t >= T_N || abs(x[3] - pi) <= 0.3
        force_LQR(x, sys_LTI_upper)
    else 
        get_control_input(x, nn_params)
    end

    # # ODE problem of nonlinear CartPole System
    tspan = (0, 10 * T_N)
    Ts = 0.001;
    t_lin = range(tspan[begin], tspan[end], step=Ts);
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f]
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇]
    prob = ODEProblem(cartPoleSystem, x0, tspan, p)
    sol = solve(prob)

    # Plot
    plot_sol(t_lin, sol)

    # make_gif(sol, sys)
    return nn_params
end


if abspath(PROGRAM_FILE) == @__FILE__
    @time main_swingup()
end
