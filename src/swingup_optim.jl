"

 author: Obrusnik Vit
"

using Ipopt
import JuMP

function swingupControlTrajectory(T_N::Float64, N::Int64,  params::CartPoleParams, x0::CartPoleState, xN::CartPoleState)
    g = 9.81
    @unpack mₜ, mₚ, bₜ, bₚ, L  = params
    x₀, ẋ₀, ϕ₀, ϕ̇₀ = unpack_state(x0)
    xₙ, ẋₙ, ϕₙ, ϕ̇ₙ = unpack_state(xN)
    
    model = JuMP.Model(Ipopt.Optimizer)

    # x vector - 4 variables
    # u -input with max/min power
    @JuMP.variable(model, x[1:4,1:N])
    @JuMP.variable(model, -100 <= u[1:N] <= 100)

    hₖ = T_N / (N-1)  # homogeneous time step between collocation points
    println("hk == $hₖ")
    @JuMP.NLobjective(model, Min, 0.5*hₖ*sum(u[i]^2 for i in 1:N))
    
    # init state
    @JuMP.NLconstraint(model, x[1,1] == x₀)
    @JuMP.NLconstraint(model, x[2,1] == ẋ₀)
    @JuMP.NLconstraint(model, x[3,1] == ϕ₀)
    @JuMP.NLconstraint(model, x[4,1] == ϕ̇₀)

    # final state
    @JuMP.NLconstraint(model, x[1,end] == xₙ)
    @JuMP.NLconstraint(model, x[2,end] == ẋₙ)
    @JuMP.NLconstraint(model, x[3,end] == ϕₙ)
    @JuMP.NLconstraint(model, x[4,end] == ϕ̇ₙ)

    for k = 1:N-1
        f_fut_1 = @JuMP.NLexpression(model, x[2,k+1])
        f_fut_2 = @JuMP.NLexpression(model, 1/(mₜ+mₚ*sin(x[3,k+1])^2) * (mₚ*sin(x[3,k+1])*(L*x[4,k+1]^2 + g*cos(x[3,k+1])) - bₜ*x[2,k+1] + (bₚ*x[4,k+1]*cos(x[3,k+1])/L) + u[k+1]))
        f_fut_3 = @JuMP.NLexpression(model, x[4,k+1])
        f_fut_4 = @JuMP.NLexpression(model, 1/(L*(mₜ+mₚ*sin(x[3,k+1])^2)) * (-mₚ*L*x[4,k+1]^2*cos(x[3,k+1])*sin(x[3,k+1]) - (mₜ+mₚ)*g*sin(x[3,k+1]) + bₜ*x[2,k+1]*cos(x[3,k+1]) - ((mₜ+mₚ)*bₚ*x[4,k+1]/(mₚ*L)) - u[k+1]*cos(x[3,k+1])))

        f_now_1 = @JuMP.NLexpression(model, x[2,k]) 
        f_now_2 = @JuMP.NLexpression(model, 1/(mₜ+mₚ*sin(x[3,k])^2) * (mₚ*sin(x[3,k])*(L*x[4,k]^2 + g*cos(x[3,k])) - bₜ*x[2,k] + (bₚ*x[4,k]*cos(x[3,k])/L) + u[k]))
        f_now_3 = @JuMP.NLexpression(model, x[4,k])
        f_now_4 = @JuMP.NLexpression(model, 1/(L*(mₜ+mₚ*sin(x[3,k])^2)) * (-mₚ*L*x[4,k]^2*cos(x[3,k])*sin(x[3,k]) - (mₜ+mₚ)*g*sin(x[3,k]) + bₜ*x[2,k]*cos(x[3,k]) - ((mₜ+mₚ)*bₚ*x[4,k]/(mₚ*L)) - u[k]*cos(x[3,k])))

        # x[k+1] - x[k] = hk/2 * (fₖ₊₁ + fₖ)
        @JuMP.NLconstraint(model, x[1,k+1]-x[1,k] == 0.5*hₖ * (f_fut_1 + f_now_1))
        @JuMP.NLconstraint(model, x[2,k+1]-x[2,k] == 0.5*hₖ * (f_fut_2 + f_now_2))
        @JuMP.NLconstraint(model, x[3,k+1]-x[3,k] == 0.5*hₖ * (f_fut_3 + f_now_3))
        @JuMP.NLconstraint(model, x[4,k+1]-x[4,k] == 0.5*hₖ * (f_fut_4 + f_now_4))
    end

    # set initial guess - very simple one, just linear progression from init state to final
    JuMP.set_start_value.(x[1,:], range(x₀, xₙ, length=N))
    JuMP.set_start_value.(x[2,:], range(ẋ₀, ẋₙ, length=N))
    JuMP.set_start_value.(x[3,:], range(ϕ₀, ϕₙ, length=N))
    JuMP.set_start_value.(x[4,:], range(ϕ̇₀, ϕ̇ₙ, length=N))

    JuMP.optimize!(model)
    @JuMP.show JuMP.termination_status(model)
    @JuMP.show JuMP.primal_status(model)
    return JuMP.value.(x), JuMP.value.(u)
end


