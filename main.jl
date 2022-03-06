"

 author: Obrusnik Vit
"

using DifferentialEquations
using Random, Distributions
using UnPack
using Ipopt
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

function plot_meas_and_est(tₘ, meas, tₒ, estim)
    p = plot(title="CartPole States (meas and estim)")
    p1 = plot(tₘ,  meas[:,1], label="x", title="cart pos")
    plot!(p1, tₒ, estim[:,1], label="x_e")
    p2 = plot(tₘ,  meas[:,2], label="xdot", title="cart speed")
    plot!(p2, tₒ, estim[:,2], label="xdot_e")
    p3 = plot(tₘ,  meas[:,3], label="phi", title="pole angle")
    plot!(p3, tₘ, estim[:,3], label="phi_e")
    p4 = plot(tₘ,  meas[:,4], label="phidot", title="pole angular speed")
    plot!(p4, tₒ, estim[:,4], label="phidot_e")
    p = plot(p1, p2, p3, p4, layout=(2, 2))
    
    display(p)
end

function unpack_sol(sol::SavedValues; indices=[1,2,3,4])
    # indices 5-8 if we have also noisy values in SavedVector
    sol_x1 = [s[indices[1]] for s in sol.saveval]
    sol_x2 = [s[indices[2]] for s in sol.saveval]
    sol_x3 = [s[indices[3]] for s in sol.saveval]
    sol_x4 = [s[indices[4]] for s in sol.saveval]
    return [sol_x1 sol_x2 sol_x3 sol_x4]
end

function unpack_sol(t, sol)
    sol_unpacked = sol.(t)
    sol_x1 = [s[1] for s in sol_unpacked]
    sol_x2 = [s[2] for s in sol_unpacked]
    sol_x3 = [s[3] for s in sol_unpacked]
    sol_x4 = [s[4] for s in sol_unpacked]
    return [sol_x1 sol_x2 sol_x3 sol_x4]
end

function corrupt_sol(t, sol_unpacked)
    sol_corrupted = zeros(size(sol_unpacked))
    for i = 1:length(t)
        noise = get_meas_noise()
        @. sol_corrupted[i, :] = sol_unpacked[i,:] + noise
    end
    return sol_corrupted 
end

function plot_with_meas_noise(t, sol_unpacked, sol_corrupted)
    p = plot(title="States (sim and meas)")
    p1 = plot(t, sol_corrupted[:,1], label="x_m", title="cart pos")
    plot!(p1, t, sol_unpacked[:,1], label="x")
    p2 = plot(t, sol_corrupted[:,2], label="xdot_m", title="cart speed")
    plot!(p2, t, sol_unpacked[:,2], label="xdot")
    p3 = plot(t, sol_corrupted[:,3], label="phi_m", title="pole angle")
    plot!(p3, t, sol_unpacked[:,3], label="phi")
    p4 = plot(t, sol_corrupted[:,4], label="phidot_m", title="pole angular speed")
    plot!(p4, t, sol_unpacked[:,4], label="phidot")
    p = plot(p1, p2, p3, p4, layout=(2, 2))
    display(p)
end


function main_just_simulate()
    # CartPole structure creation
    params = CartPoleParams();
    init = CartPoleState(0.0, 0.0, 0.0, 0.0);
    # sys = CartPole(params, init);

    # Time
    tspan = (0.0, 20.0);
    Ts = 0.001;
    t_lin = range(tspan[begin], tspan[end], step=Ts);

    # # Forcing function
    # f(x,t) = f_step(x, t; t_step_begin=9.3, t_step_end=11);
    f(x, t) = f_step(x, t; t_step_begin=4, t_step_end=6);

    # # ODE problem of nonlinear CartPole System
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f];
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇];
    prob = ODEProblem(cartPoleSystem, x0, tspan, p);
    sol = solve(prob);

    # Plot
    sol_unpacked = unpack_sol(t_lin, sol)
    p1 = plot(t_lin, sol_unpacked[:,1], label=L"x")
    p2 = plot(t_lin, sol_unpacked[:,2], label=L"\dot{x}")
    p3 = plot(t_lin, sol_unpacked[:,3], label=L"\phi")
    p4 = plot(t_lin, sol_unpacked[:,4], label=L"\dot{\phi}")
    p = plot(p1, p2, p3, p4, layout=(2, 2))
    display(p)
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
    f(x, t) = f_step(x, t; t_step_begin=4, t_step_end=6);

    # Callback for observer
    sys_obs = cartpole_observer(sys_LTI);
    sysd_obs = c2d(sys_obs, Ts, :zoh);
    u_obs = [0.0; 0.0; 0.1; 0.0];
    function cb_observer(u, t)
        noise = get_meas_noise()
        u[1] += noise
        # u[1] or sys_LTI_lower.C*u
        u_obs = sysd_obs.A * [u_obs[1]; u_obs[2]; u_obs[3]; u_obs[4]] + sysd_obs.B * [f(u, t); sys_LTI.C * u];
        return vcat(u_obs, u);
    end

    # Proper LKF
    P = 0.001   * I(4);
    Q = 0.00001 * I(4);
    R = 0.1 * I(1);

    A = sysd_LTI.A;
    B = sysd_LTI.B;
    C = sysd_LTI.C; 
    x = [0.0; 0.0; pi - 0.1; 0.0];

    function cb_LKF(u, t, _)
        noise = get_meas_noise()
        u[1] += noise  # measurement x is noisy

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
        return vcat(x_estim, u);
    end

    function cb_LKF_2(u, t, _)
        noise = get_meas_noise()
        u[1] += noise

        # Time update / Prediction step / Time step
        xₖ = A * x + B * f(u, t)
        Pₖ = A * P * A' + Q

        # Meas update / Correction step / Data step
        y = C * u
        Lₖ = (Pₖ * C') / (C * Pₖ * C' + R)
        x = xₖ + Lₖ * (y - C * xₖ);
        P = Pₖ - Lₖ * (C * Pₖ * C') * Lₖ'

        x_estim = vec(x);
        return vcat(x_estim, u);
    end

    function cb_EKF(u, t, params)
        noise = get_meas_noise()
        u[1] += noise  # measurement x is noisy

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
        x_sol = solve(ODEProblem(cartPoleSystem, xₖ, (t, t + Ts), params))
        x = x_sol.u[end]

        P = Aₖ * Pₖ * Aₖ' + Q

        x_estim = vec(x);
        return vcat(x_estim, u);
    end

    # # ODE problem of nonlinear CartPole System
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f];
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇];
    prob = ODEProblem(cartPoleSystem, x0, tspan, p);
    saved_values = SavedValues(Float64, Vector{Float64});
    cb = SavingCallback((u, t, integrator) -> cb_EKF(u, t, p), saved_values; saveat=t_lin);
    sol = solve(prob, callback=cb);

    # Plot
    sol_unpacked = unpack_sol(t_lin, sol)
    estim_unpack = unpack_sol(saved_values, indices=[1, 2, 3, 4])
    sol_unpacked_noisy = unpack_sol(saved_values, indices=[5, 6, 7, 8])
    p1 = plot(t_lin, sol_unpacked_noisy[:,1], label=L"x_{meas}", alpha=0.3)
    plot!(p1, t_lin, sol_unpacked[:,1], label=L"x_{true}", linewidth=1.5)
    plot!(p1, t_lin, estim_unpack[:,1], label=L"x_{estim}", style=:dash, linewidth=1.5)
    p2 = plot(t_lin, sol_unpacked_noisy[:,2], label=L"\dot{x}_{meas}", alpha=0.3)
    plot!(p2, t_lin, sol_unpacked[:,2], label=L"\dot{x}_{true}", linewidth=1.5)
    plot!(p2, t_lin, estim_unpack[:,2], label=L"\dot{x}_{estim}", style=:dash, linewidth=1.5)
    p3 = plot(t_lin, sol_unpacked_noisy[:,3], label=L"\phi_{meas}", alpha=0.3)
    plot!(p3, t_lin, sol_unpacked[:,3], label=L"\phi_{true}", linewidth=1.5)
    plot!(p3, t_lin, estim_unpack[:,3], label=L"\phi_{estim}", style=:dash, linewidth=1.5)
    p4 = plot(t_lin, sol_unpacked_noisy[:,4], label=L"\dot{\phi}_{meas}", alpha=0.3)
    plot!(p4, t_lin, sol_unpacked[:,4], label=L"\dot{\phi}_{true}", linewidth=1.5)
    plot!(p4, t_lin, estim_unpack[:,4], label=L"\dot{\phi}_{estim}", style=:dash, linewidth=1.5)
    p = plot(p1, p2, p3, p4, layout=(2, 2))
    display(p)
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
    sol_unpacked = unpack_sol(t_lin, sol)
    p = plot(t_lin, sol_unpacked)
    display(p)
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
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f]
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇]
    prob = ODEProblem(cartPoleSystem, x0, tspan, p)
    sol = solve(prob)
    # plot(sol)

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
    p = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f]
    x0 = [init.x, init.ẋ, init.ϕ, init.ϕ̇]
    prob = ODEProblem(cartPoleSystem, x0, tspan, p)
    sol = solve(prob)
    plot(sol)

    make_gif(sol, sys)
    return nn_params
end


if abspath(PROGRAM_FILE) == @__FILE__
    @time main_swingup()
end
