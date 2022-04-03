"

 author: Obrusnik Vit
"

function unpack_sol(sol::SavedValues; indices=[1,2,3,4])
    # indices 1-4 for true solution
    # indices 5-8 if we have also noisy values in SavedVector
    # index 9 for force
    ret = []
    for i = indices
        sol_unpack = [s[i] for s in sol.saveval]
        push!(ret, sol_unpack)
    end
    return ret
end

function unpack_sol(t, sol)
    sol_unpacked = sol.(t)
    ret = []
    for i = [1, 2, 3, 4]
        sol_x = [s[i] for s in sol_unpacked]
        push!(ret, sol_x)
    end
    return ret
end

function corrupt_sol(t, sol_unpacked)
    sol_corrupted = zeros(size(sol_unpacked))
    for i = 1:length(t)
        noise = get_meas_noise()
        @. sol_corrupted[i, :] = sol_unpacked[i,:] + noise
    end
    return sol_corrupted 
end

function plot_sol(sol; dest_dir="output", dest_name="plot_sol.png")
    t_lin = range(sol.t[begin], sol.t[end], step=0.1)
    sol_unpacked = unpack_sol(t_lin, sol)
    p1 = plot(t_lin, sol_unpacked[1], lab=L"x(t)", xlab=L"time [s]", ylab=L"x [m]")
    p2 = plot(t_lin, sol_unpacked[2], lab=L"\dot{x}(t)", xlab=L"time [s]", ylab=L"\dot{x} [m/s]")
    p3 = plot(t_lin, sol_unpacked[3], lab=L"\phi(t)", xlab=L"time [s]", ylab=L"\phi [rad]")
    p4 = plot(t_lin, sol_unpacked[4], lab=L"\dot{\phi}(t)", xlab=L"time [s]", ylab=L"\dot{\phi} [rad/s]")
    p = plot(p1, p2, p3, p4, layout=(2, 2))
    savefig(p, dest_dir * "/" * dest_name)
    display(p)
end

function plot_sol_force(sol, saved_values; dest_dir="output", dest_name="plot_sol_force.png")
    t_lin = saved_values.t
    sol_unpacked = unpack_sol(t_lin, sol)
    force_unpack = unpack_sol(saved_values, indices=[5])

    p1 = plot(t_lin, sol_unpacked[1], lab=L"x(t)", lw=1.5, xlab=L"time [s]", ylab=L"x [m]")
    p2 = plot(t_lin, sol_unpacked[2], lab=L"\dot{x}(t)", lw=1.5, xlab=L"time [s]", ylab=L"\dot{x} [m/s]")
    p3 = plot(t_lin, sol_unpacked[3], lab=L"\phi(t)", lw=1.5, xlab=L"time [s]", ylab=L"\phi [rad]")
    p4 = plot(t_lin, sol_unpacked[4], lab=L"\dot{\phi}(t)", lw=1.5, xlab=L"time [s]", ylab=L"\dot{\phi} [rad/s]")

    # force
    p5 = plot(t_lin, force_unpack[1], lab=L"f(x,t)", lw=1.5, xlab=L"time [s]", ylab=L"F[N]")
    empty_plot = plot([0],[0], frame=:none, ticks=:none, lab=:none)

    p = plot(p1, p2, p3, p4, p5, empty_plot, layout=(3, 2))
    savefig(p, dest_dir * "/" * dest_name)
    display(p)
end


function plot_sol_est_force(sol, saved_values; dest_dir="output", dest_name="plot_sol_est_force.png")
    # we measure only `x` so other "meas" plots are commented out
    t_lin = saved_values.t
    sol_unpacked = unpack_sol(t_lin, sol)
    estim_unpack = unpack_sol(saved_values, indices=[1, 2, 3, 4])
    force_unpack = unpack_sol(saved_values, indices=[9])

    sol_unpacked_noisy = unpack_sol(saved_values, indices=[5, 6, 7, 8])
    p1 = plot(t_lin, sol_unpacked_noisy[1], lab=L"x_{meas}(t)", alpha=0.3, xlab=L"time [s]", ylab=L"x [m]")
    plot!(p1, t_lin, sol_unpacked[1], lab=L"x_{true}(t)", lw=1.5)
    plot!(p1, t_lin, estim_unpack[1], lab=L"x_{estim}(t)", style=:dash, lw=1.5)
    # p2 = plot(t_lin, sol_unpacked_noisy[2], lab=L"\dot{x}_{meas}(t)", alpha=0.3, xlab=L"time [s]", ylab=L"\dot{x} [m/s]")
    p2 = plot(t_lin, sol_unpacked[2], lab=L"\dot{x}_{true}(t)", lw=1.5, xlab=L"time [s]", ylab=L"\dot{x} [m/s]")
    plot!(p2, t_lin, estim_unpack[2], lab=L"\dot{x}_{estim}(t)", style=:dash, lw=1.5)
    # p3 = plot(t_lin, sol_unpacked_noisy[3], lab=L"\phi_{meas}(t)", alpha=0.3, xlab=L"time [s]", ylab=L"\phi [rad]")
    p3 = plot(t_lin, sol_unpacked[3], lab=L"\phi_{true}(t)", lw=1.5, xlab=L"time [s]", ylab=L"\phi [rad]")
    plot!(p3, t_lin, estim_unpack[3], lab=L"\phi_{estim}(t)", style=:dash, lw=1.5)
    # p4 = plot(t_lin, sol_unpacked_noisy[4], lab=L"\dot{\phi}_{meas}(t)", alpha=0.3, xlab=L"time [s]", ylab=L"\dot{\phi} [rand/s]")
    p4 = plot(t_lin, sol_unpacked[4], lab=L"\dot{\phi}_{true}(t)", lw=1.5, xlab=L"time [s]", ylab=L"\dot{\phi} [rad/s]")
    plot!(p4, t_lin, estim_unpack[4], lab=L"\dot{\phi}_{estim}(t)", style=:dash, lw=1.5)

    # force
    p5 = plot(t_lin, force_unpack[1], lab=L"f(x,t)", lw=1.5, xlab=L"time [s]", ylab=L"F[N]")
    empty_plot = plot([0],[0], frame=:none, ticks=:none, lab=:none)

    p = plot(p1, p2, p3, p4, p5, empty_plot, layout=(3, 2))
    savefig(p, dest_dir * "/" * dest_name)
    display(p)
end


function vis_cartpole(cartpole::CartPole, time::Float64)
    function circle_shape(h, k, r)
        Θ = LinRange(0, 2 * π, 25)
        h .+ r * sin.(Θ), k .+ r * cos.(Θ)
    end
    xlims = (-6, 6)
    ylims = (-3, 4)
    p = plot(title = "Cartpole",
        xlims = xlims, ylims = ylims, xlabel = "x[m]", ylabel = "y[m]", aspect_ratio = :equal,
        annotation = (xlims[1], ylims[2], Plots.text("t=$time", :left, :top))
    )
    hline!(p, [0.0], color=:black, alpha=0.4)

    @unpack w, h, L = cartpole.params
    @unpack x, ϕ = cartpole.state

    # cart
    wheel_radius = h / 5
    cart = Shape([(x - w / 2, wheel_radius), (x + w / 2, wheel_radius), (x + w / 2, h), (x - w / 2, h), (x - w / 2, wheel_radius)])
    plot!(p, cart, label = "", color = :white, fillcolor = :lightgrey)
    # scatter!(p, [x-w/3, x+w/3], [0.0, 0.0], label="", markersize=10*h, alignment=:north, color=:black)
    plot!(p, circle_shape(x - w / 3, wheel_radius, wheel_radius), seriestype = [:shape,], lw = 0.5, c = :black, linecolor = :black, legend = false, fillalpha = 1.0, aspect_ratio = 1)
    plot!(p, circle_shape(x + w / 3, wheel_radius, wheel_radius), seriestype = [:shape,], lw = 0.5, c = :black, linecolor = :black, legend = false, fillalpha = 1.0, aspect_ratio = 1)

    # pole
    tx = x + L * sin(ϕ)
    ty = h / 2 - L * cos(ϕ)
    plot!(p, [(x, h / 2), (tx, ty)], linewidth = 3, label = "", color = :black)
    # display(p)
end


function make_gif(t, states, cartpole::CartPole)
    # was used for swing up control vis
    p = plot(reuse = false)
    anim = @animate for idx = 1:1:length(t)
        tim = t[idx]
        x = states[1, idx]
        ẋ = states[2, idx]
        ϕ = states[3, idx]
        ϕ̇ = states[4, idx]
        cartpole.state = CartPoleState(x, ẋ, ϕ, ϕ̇)
        vis_cartpole(cartpole, tim)
    end
    gif(anim,dest_dir * "/" * "cart_pole_sim.gif", fps = 24)
end


function make_gif(sol, cartpole::CartPole; dest_dir::String="output", dest_name::String="cart_pole_sim.gif")
    # p = plot(reuse = false)
    anim = @animate for t = sol.t[begin]:0.1:sol.t[end]
        v = sol(t)
        x = v[1]
        ẋ = v[2]
        ϕ = v[3]
        ϕ̇ = v[4]
        cartpole.state = CartPoleState(x, ẋ, ϕ, ϕ̇)
        vis_cartpole(cartpole, t)
    end
    gif(anim, dest_dir * "/" * dest_name, fps = 24)
end
