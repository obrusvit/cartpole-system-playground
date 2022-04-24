"

 author: Obrusnik Vit
"


"""
Representation of parameters in CartPole system. Unmutable.
"""
mutable struct CartPoleParams
    w 
    h
    mₜ
    mₚ 
    L
    bₜ
    bₚ
end


"""
Default constructor for CartPoleParams.
"""
CartPoleParams() = CartPoleParams(
        1.0, 0.5,     # w, h
        1.0, 0.1,   # mt, mp
        2.0,        # L
        0.5, 0.1  # bt, bp
    )


#-------------------------------------------------------------------------------

"""
Representation of state in CartPole system. Mutable.
"""
mutable struct CartPoleState
    x
    ẋ   
    ϕ
    ϕ̇
end


"""
Default constructor for CartPoleState.
Initializes all states to 0.
"""
CartPoleState() = CartPoleState(0.0, 0.0, 0.0, 0.0)


"""
Constructor for CartPoleState.
Initializes pole angle to given angle. Other states are initialized to 0.
CartPoleState(pi) is convenience for inverse cartpole control.
"""
CartPoleState(poleAngleRad) = CartPoleState(0.0, 0.0, poleAngleRad, 0.0)



function unpack_state(s::CartPoleState)
    return s.x, s.ẋ, s.ϕ, s.ϕ̇
end

#-------------------------------------------------------------------------------

mutable struct CartPole
    params::CartPoleParams
    state::CartPoleState
end
CartPole() = CartPole(CartPoleParams(), CartPoleState())


#-------------------------------------------------------------------------------

function cartPoleSystem(du, u, p, t)
    # To be used with DifferentialEquations package
    # State: x, xdot, phi, phidot
    # With damping on both: cart and pole
    g  = 9.81;
    mₜ, mₚ, L, bₜ, bₚ, F  = p  # function f(x,t)
    x, ẋ, ϕ, ϕ̇ = u
    force = F(u, t)

    du[1] = ẋ 
    du[2] = 1/(mₜ+mₚ*sin(ϕ)^2) * (mₚ*sin(ϕ)*(L*ϕ̇^2 + g*cos(ϕ)) - bₜ*ẋ + (bₚ*ϕ̇*cos(ϕ)/L) + force )
    du[3] = ϕ̇
    du[4] = 1/(L*(mₜ+mₚ*sin(ϕ)^2)) * (-mₚ*L*ϕ̇^2*cos(ϕ)*sin(ϕ) - (mₜ+mₚ)*g*sin(ϕ) + bₜ*ẋ*cos(ϕ) - ((mₜ+mₚ)*bₚ*ϕ̇/(mₚ*L)) - force*cos(ϕ))
end

function cartPoleDiscApprox(cartpole::CartPole, force, w, Dt)
    # Discrete Euler approximation of the above equations
    g = 9.81
    @unpack mₜ, mₚ, bₜ, bₚ, L  = cartpole.params

    state = cartpole.state
    x = state.x
    ẋ = state.ẋ
    ϕ = state.ϕ
    ϕ̇ = state.ϕ̇

    x_new = x + Dt * ẋ 
    ẋ_new = ẋ + Dt * (1/(mₜ+mₚ*sin(ϕ)^2)) * (mₚ*sin(ϕ)*(L*ϕ̇^2 + g*cos(ϕ)) - bₜ*ẋ + (bₚ*ϕ̇*cos(ϕ)/L) + force )
    ϕ_new = ϕ + Dt * ϕ̇
    ϕ̇_new = ϕ̇ + Dt * (1/(L*(mₜ+mₚ*sin(ϕ)^2))) * (-mₚ*L*ϕ̇^2*cos(ϕ)*sin(ϕ) - (mₜ+mₚ)*g*sin(ϕ) + bₜ*ẋ*cos(ϕ) - ((mₜ+mₚ)*bₚ*ϕ̇/(mₚ*L)) - force*cos(ϕ))
    return [x_new; ẋ_new; ϕ_new; ϕ̇_new]
end


cartpoleJacobian(cartpole::CartPole) = cartpoleJacobian(cartpole.params, cartpole.state, 0.0)
cartpoleJacobian(cartpole::CartPole, F₀) = cartpoleJacobian(cartpole.params, cartpole.state, F₀)
cartpoleJacobian(params::CartPoleParams, state::CartPoleState) = cartpoleJacobian(params, state, 0.0)
"""
The function takes linearization around state and returns matrices A,B - jacobian matrices evaluated at the point state.
"""
function cartpoleJacobian(params::CartPoleParams, state::CartPoleState, F₀)
    g = 9.81
    @unpack mₜ, mₚ, bₜ, bₚ, L  = params
    x₀, ẋ₀, ϕ₀, ϕ̇₀ = unpack_state(state)

    @Symbolics.variables x ẋ ϕ ϕ̇ F
    cartpole_equations = [
        ẋ,
        1/(mₜ+mₚ*sin(ϕ)^2) * (mₚ*sin(ϕ)*(L*ϕ̇^2 + g*cos(ϕ)) - bₜ*ẋ + (bₚ*ϕ̇*cos(ϕ)/L) + F ),
        ϕ̇,
        1/(L*(mₜ+mₚ*sin(ϕ)^2)) * (-mₚ*L*ϕ̇^2*cos(ϕ)*sin(ϕ) - (mₜ+mₚ)*g*sin(ϕ) + bₜ*ẋ*cos(ϕ) - ((mₜ+mₚ)*bₚ*ϕ̇/(mₚ*L)) - F*cos(ϕ))
    ]

    A = Symbolics.jacobian(cartpole_equations, [x, ẋ, ϕ, ϕ̇])
    A_ret = Symbolics.substitute.(A, (Dict(x=>x₀, ẋ=>ẋ₀, ϕ=>ϕ₀, ϕ̇=>ϕ̇₀, F=>0.0),))

    B = Symbolics.jacobian(cartpole_equations, [F])
    B_ret = Symbolics.substitute.(B, (Dict(x=>x₀, ẋ=>ẋ₀, ϕ=>ϕ₀, ϕ̇=>ϕ̇₀, F=>F₀),))

    # from Matrix{Num} to Matrix{Real} to Matrix{Float64}
    return Float64.(Symbolics.value.(A_ret)), Float64.(Symbolics.value.(B_ret))  
end


cartpole_LTI_sys(cartpole::CartPole) = cartpole_LTI_sys(cartpole, cartpole.state, 0.0)
cartpole_LTI_sys(cartpole::CartPole, force) = cartpole_LTI_sys(cartpole, cartpole.state, force)
cartpole_LTI_sys(cartpole::CartPole, state::CartPoleState) = cartpole_LTI_sys(cartpole, state, 0.0)
function cartpole_LTI_sys(cartpole::CartPole, init_state::CartPoleState, force)
    A, B = cartpoleJacobian(cartpole.params, init_state, force)
    # C = I(4)
    C = [1 0 0 0]
    D = 0;
    ss(A, B, C, D)
end

function cartpole_observer_subsys(sys_lti)
    # when we don't care about the position x
    A = sys_lti.A[2:end, 2:end]
    B = sys_lti.B[2:end]
    C = [1 0 0]
    D = 0

    # L = place(A', C', [-5+3im -5-3im -1])
    L = place(A', C', [-5 -5 -1])
    L = L'
    Aₒ = A - L*C
    Bₒ = [B L]
    Cₒ = I(3)
    Dₒ = 0
    return ss(Aₒ, Bₒ, Cₒ, Dₒ)
end

function cartpole_observer(sys_lti)
    A = sys_lti.A
    B = sys_lti.B
    C = sys_lti.C
    D = sys_lti.D

    # L = place(A', C', [-1.0 -1.1 -1.2 -1.3])
    # L = L'
    L = kalman(A, C, 0.001*I(4), 0.1)
    Aₒ = A - L*C
    Bₒ = [B L]
    Cₒ = I(4)
    Dₒ = 0
    return ss(Aₒ, Bₒ, Cₒ, Dₒ)
end
