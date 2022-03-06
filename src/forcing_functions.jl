"

 author: Obrusnik Vit
"

f0(x, t) = 0  # no force

f_ramp(x, t; c=0.1) = c*t

f_step(x, t; t_step_begin=10, t_step_end=11) = (t>= t_step_begin && t <= t_step_end) ? 1 : 0 

function forcing_1(x, t, L_gain)
    ret = (-L_gain*(x .- [0; 0; pi; 0]))[1]
    if t >=10 && t<=11 
        ret += 10
    end
    ret
end


# The function used during swingup
function force_swingup_optim(u, t, T_N)
    N = length(u)
    p = t/T_N  # portion, in range [0,1]
    idx = Int(floor(p*N))
    # safety measures, should be handled better.
    if idx==0 idx=1 end
    u[idx]     
end

# The function used to stabilize in upper position
# Q = [0.1 0    0   0; 
#      0   0.01 0   0;
#      0   0    100   0;
#      0   0    0   0.1]
function force_LQR(x, LTI_sys)
    Q = [0.1 0    0   0; 
         0   0.01  0   0;
         0   0    100  0;
         0   0    0   0.1]
    L_gain = lqr(LTI_sys, Q, 1.0*I)
    (-L_gain*(x .- [0; 0; pi; 0]))[1]
end



