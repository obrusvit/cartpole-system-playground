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
    Q_true = [0.001 0.0 0.0 0.0
        0.0 0.001 0.0 0.0
        0.0 0.0 0.001 0.0
        0.0 0.0 0.0 0.001]
    dist = MvNormal(means, Q_true)
    return rand(dist)
end


function cb_save(u, t, p)
    f = p[end]
    return vcat(u, f(u, t))
end

function simulate_cartpole(cp_params::CartPoleParams, init_state::CartPoleState, forcing_function, tspan::Tuple{Number,Number}, Ts::Float64; saving_callback=cb_save)
    f(x, t) = forcing_function(x, t)
    # # ODE problem of nonlinear CartPole System
    p = [cp_params.mₜ, cp_params.mₚ, cp_params.L, cp_params.bₜ, cp_params.bₚ, f]
    x0 = [init_state.x, init_state.ẋ, init_state.ϕ, init_state.ϕ̇]
    prob = ODEProblem(cartPoleSystem, x0, tspan, p)
    saved_values = SavedValues(Float64, Vector{Float64})
    cb = SavingCallback((u, t, integrator) -> saving_callback(u, t, p), saved_values; saveat = range(tspan[begin], tspan[end], step = Ts))
    sol = DifferentialEquations.solve(prob, callback = cb)

    return sol, saved_values
end


function main_estimator_cb(cp_params::CartPoleParams, init_state::CartPoleState; make_plot::Bool=true, make_gif::Bool=false)
    Ts = 0.01

    # # Forcing function
    forcing_fun(x, t) = f_step(x, t; t_step_begin = 4, t_step_end = 6)

    # state to be propagated in estimator - no error
    x = [init_state.x; init_state.ẋ; init_state.ϕ; init_state.ϕ̇]

    # LTI at lower pos
    sys = CartPole(cp_params, init_state)
    sys_LTI = cartpole_LTI_sys(sys, CartPoleState(0, 0, 0, 0))
    sysd_LTI = c2d(sys_LTI, Ts, :zoh)

    # Callback for observer
    sys_obs = cartpole_observer(sys_LTI)
    sysd_obs = c2d(sys_obs, Ts, :zoh)

    function cb_observer(u, t, p)
        noise = get_meas_noise()
        u[1] += noise
        f = p[end]
        # u[1] or sys_LTI_lower.C*u
        x = sysd_obs.A * [x[1]; x[2]; x[3]; x[4]] + sysd_obs.B * [f(u, t); sys_LTI.C * u]
        return vcat(x, u, f(u, t))
    end

    # Proper LKF
    P = 0.01 * I(4)
    Q = 0.0001 * I(4)
    R = 0.1 * I(1)

    A = sysd_LTI.A
    B = sysd_LTI.B
    C = sysd_LTI.C

    function cb_LKF(u, t, p)
        noise = get_meas_noise()
        u[1] += noise  # measurement x is noisy
        f = p[end]

        # Meas update / Correction step / Data step
        y = C * u
        Lₖ = (P * C') / (C * P * C' + R)
        xₖ = x + Lₖ * (y - C * x)
        # Pₖ = P - Lₖ*(C*P*C')*Lₖ'
        Pₖ = P - Lₖ * C * P

        # Time update / Prediction step / Time step
        x = A * xₖ + B * f(u, t)
        P = A * Pₖ * A' + Q

        x_estim = vec(x)
        return vcat(x_estim, u, f(u, t))
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
        x = xₖ + Lₖ * (y - C * xₖ)
        P = Pₖ - Lₖ * (C * Pₖ * C') * Lₖ'

        x_estim = vec(x)
        return vcat(x_estim, u, f(u, t))
    end

    function cb_EKF(u, t, p)
        noise = get_meas_noise()
        u[1] += noise  # measurement x is noisy
        f = p[end]

        # Meas update / Correction step / Data step
        y = C * u
        Lₖ = (P * C') / (C * P * C' + R)
        xₖ = x + Lₖ * (y - C * x)
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

        x_estim = vec(x)
        return vcat(x_estim, u, f(u, t))
    end

    sol, saved_values = simulate_cartpole(cp_params, init_state, forcing_fun, (0.0, 20.0), Ts, saving_callback=cb_EKF)
    # Plot
    if make_plot
        plot_sol_est_force(sol, saved_values)
    end
    if make_gif
        make_gif(sol, CartPole(cp_params, init_state); dest_dir="output", dest_name="main_lqr.gif")
    end
end

function main_LQR(cp_params::CartPoleParams, init_state::CartPoleState; make_plot::Bool=true, make_gif::Bool=false)
    # LTI at upper pos
    sys = CartPole(cp_params, init_state)
    sys_LTI_upper = cartpole_LTI_sys(sys, CartPoleState(0, 0, pi, 0))

    # # Forcing function
    Q = [0.1 0 0 0
        0 0.01 0 0
        0 0 100 0
        0 0 0 0.1]
    L_gain = lqr(sys_LTI_upper, Q, 0.1 * I)
    f(x, t) = forcing_1(x, t, L_gain)

    sol, saved_values = simulate_cartpole(cp_params, init_state, f, (0.0, 20.0), 0.01)
    if make_plot
        plot_sol_force(sol, saved_values; dest_dir="output", dest_name="main_LQR.png")
    end
    if make_gif
        gif_sol(sol, CartPole(cp_params, init_state); dest_dir="output", dest_name="main_lqr.gif")
    end
end


function main_swingup_optim(cp_params::CartPoleParams, init_state::CartPoleState; make_plot::Bool=true, make_gif::Bool=false)
    final = CartPoleState(pi)
    sys = CartPole(cp_params, init_state)
    sys_LTI_upper = cartpole_LTI_sys(sys, final)

    T_N = 4.0  # final time of the maneuver
    N = 1001     # number of collocation points
    _, u = swingupControlTrajectory(T_N, N, cp_params, init_state, final)

    f(x, t) =
        if t >= T_N || abs(x[3] - pi) <= 0.3
            force_LQR(x, sys_LTI_upper)
        else
            force_swingup_optim(u, t, T_N)
        end

    sol, saved_values = simulate_cartpole(cp_params, init_state, f, (0.0, 3*T_N), 0.01)
    if make_plot
        plot_sol_force(sol, saved_values; dest_dir="output", dest_name="main_swingup_optim.png")
    end
    if make_gif
        gif_sol(sol, CartPole(cp_params, init_state); dest_dir="output", dest_name="main_swingup_optim.gif")
    end
end


function main_swingup_rl(cp_params::CartPoleParams, init_state::CartPoleState; make_plot::Bool=true, make_gif::Bool=false, nn_params::Vector{Float32}=Vector{Float32}())
    final = CartPoleState(pi)
    sys = CartPole(cp_params, init_state)
    sys_LTI_upper = cartpole_LTI_sys(sys, final)

    T_N = 4.0  # final time of the maneuver
    N = 401
    if isempty(nn_params)
        nn_params = train_cartpole_rl_controller(T_N, N, cp_params, init_state, final; saveToJson = true)
        println("training finished")
    end

    f(x, t) =
        if t >= T_N || abs(x[3] - pi) <= 0.1
            force_LQR(x, sys_LTI_upper)
        else
            get_control_input(x, nn_params)
        end

    sol, saved_values = simulate_cartpole(cp_params, init_state, f, (0.0, 3*T_N), 0.01)

    if make_plot
        plot_sol_force(sol, saved_values; dest_dir="output", dest_name="main_swingup_rl.png")
    end
    if make_gif
        gif_sol(sol, CartPole(cp_params, init_state); dest_dir="output", dest_name="main_swingup_rl.gif")
    end
    return nn_params
end


function main()
    cp_params = CartPoleParams(
        1.0, 0.5,     # w, h
        1.0, 0.3,   # mt, mp
        1.0,        # L
        0.2, 0.2  # bt, bp
    )

    init = CartPoleState(0.0, 0.0, pi - 0.3, 0.0)
    f(x, t) = f_step(x, t; t_step_begin = 4, t_step_end = 6, weight = 0.3)
    sol, saved_values = simulate_cartpole(cp_params, init, f, (0.0, 20.0), 0.01)

    # Plot
    plot_sol_force(sol, saved_values; dest_name = "main.png")
    sys = CartPole(cp_params, init)
    gif_sol(sol, sys; dest_name = "main.gif")
end
