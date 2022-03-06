"

 author: Obrusnik Vit
"

function vis_cartpole(cartpole::CartPole, time::Float64)
    xlims = (-10, 10)
    ylims = (-3, 4)
    p = plot(title="Cartpole", 
             xlims=xlims, ylims=ylims, aspect_ratio=:equal, 
             annotation=(xlims[1], ylims[2], Plots.text("t=$time", :left, :top))
            )

    @unpack w, h, L = cartpole.params
    @unpack x, ϕ = cartpole.state

    cart = Shape([(x-w/2, 0), (x+w/2, 0), (x+w/2, h), (x-w/2, h)]) 
    plot!(p, cart, label="")

    tx =   x + L*sin(ϕ)
    ty = h/2 - L*cos(ϕ)

    plot!(p, [(x, h/2), (tx, ty)], linewidth=3, label="")
    # display(p)
end


function make_gif(t, states, cartpole::CartPole)
    # was used for swing up control vis
    anim = @animate for idx = 1:1:length(t)
        tim = t[idx]
        x = states[1, idx]
        ẋ = states[2, idx]
        ϕ = states[3, idx]
        ϕ̇ = states[4, idx]
        cartpole.state = CartPoleState(x, ẋ, ϕ, ϕ̇)
        vis_cartpole(cartpole, tim)
    end
    gif(anim, "cart_pole_sim.gif", fps=24)
end


function make_gif(sol, cartpole::CartPole)
    anim = @animate for t = sol.t[begin]:0.1:sol.t[end]
        v = sol(t)
        x = v[1]
        ẋ = v[2]
        ϕ = v[3]
        ϕ̇ = v[4]
        cartpole.state = CartPoleState(x, ẋ, ϕ, ϕ̇)
        vis_cartpole(cartpole, t)
    end
    gif(anim, "cart_pole_sim.gif", fps=24)
end
