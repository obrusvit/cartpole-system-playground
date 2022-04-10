using DiffEqFlux
using DifferentialEquations

using JSON
using Dates

include("diffeq_simulation.jl")


# controller = FastChain((x, p) -> x, FastDense(3, 3, relu), FastDense(3,3,relu), FastDense(3,1))  # good
controller = FastChain((x, p) -> x, FastDense(4, 4, relu), FastDense(4,4,relu), FastDense(4,1))  
# controller = FastChain((x, p) -> x, FastDense(5, 5, relu), FastDense(5,5,relu), FastDense(5,1))  

# get_control_input(u, nn_params) = controller([cos(u[3]), sin(u[3]), u[4]], nn_params)[1]
get_control_input(u, nn_params) = controller([u[1], cos(u[3]), sin(u[3]), u[4]], nn_params)[1]
# get_control_input(u, nn_params) = controller([u[1], u[2], cos(u[3]), sin(u[3]), u[4]], nn_params)[1]

# map angle to [-pi, pi)
modpi(theta) = mod2pi(theta + pi) - pi

function read_nn_params_json(filepath::String)
    file = open(filepath)
    nn_params = JSON.parse(readline(file))
    return Vector{Float32}(nn_params)
end

function train_cartpole_rl_controller(T_N::Float64, N::Int64, params::CartPoleParams, x0::CartPoleState, xN::CartPoleState; saveToJson::Bool=false)
    # initial condition
    # u0, tspan, N, tsteps, dt = get_simulation_params()
    u0 = [x0.x, x0.ẋ, x0.ϕ, x0.ϕ̇]
    tspan = (0.0, T_N)
    tsteps = range(tspan[1], length=N, tspan[2])

    nn_pinit = initial_params(controller)
    f(x,t) = get_control_input(x, nn_pinit)
    ode_params = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f]

    # set up ODE problem
    prob = ODEProblem(cartPoleSystem, u0, tspan, ode_params)

    function loss_neuralode(p)
        f(x,t) = get_control_input(x, p)
        ode_params = [params.mₜ, params.mₚ, params.L, params.bₜ, params.bₚ, f]
        sol = solve(remake(prob, p=ode_params), Tsit5(), saveat = tsteps)
        x = sol[1, :]
        dx = sol[2, :]
        theta = modpi.(sol[3, :])
        # theta = sol[3, :]
        dtheta = sol[4, :]

        state_vec = [[u[1], u[2], u[3], u[4]] for u in sol.u] 
        force = [get_control_input(u,p) for u in state_vec]

        # good objective functions
        # loss = 100*(theta[end]-pi)^2 + dtheta[end]^2 + dx[end]^2 + 0.01 * sum(abs2, force) / N  # best so far with tspan=(0,1), length of pole=1, 

        loss = 100*(theta[end]-pi)^2 + dtheta[end]^2 + 50*x[end]^2 + dx[end]^2 + 0.01 * sum(abs2, force) / N  

        # loss = 1000*(theta[end]-pi)^2 + 10*sum(abs2, x) / N + dtheta[end]^2 + dx[end]^2 + 0.1 * sum(abs2, force) / N   # big dtheta at the end

        # loss = 10*(theta[end]-pi)^2 + dtheta[end]^2 + 0.01 * sum(abs2, force) / N  

        return loss, sol
    end

    # Training
    i = 0 # training epoch 
    # callback function after each training epoch
    # p = parameter of NN
    # l = loss
    # pred = sol
    callback_1 = function (p, l, pred; doplot=true)
        i += 1
        println("epoch: $i, loss: $l")
        if i % 100 == 0 && doplot
            p = plot(pred, label=[L"x(t)" L"\dot{x}(t)" L"\theta(t)" L"\dot{\theta}(t)"])
            display(p)
        end
        return false
    end
    println("Training starts..")
    result = DiffEqFlux.sciml_train(
      loss_neuralode,
      nn_pinit,
      # ADAM(0.05),
      cb=callback_1,
      maxiters=2400,
    )

    if saveToJson
        dtime_str = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
        open(io -> write(io, json(result)), "assets/rl_ctrl_" * dtime_str * ".json", "w")
    end
    return result
end
