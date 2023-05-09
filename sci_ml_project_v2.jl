using CSV, DataFrames, DelimitedFiles, LinearAlgebra, GeoStats, KrigingEstimators, Variography, Random, StatsBase, Trapz, CubicSplines, Plots
cd("D:\\Job\\Courses\\6_337 Parallel Computing and Scientific Machine Learning/Project/")

################## Reading the Data ############################

D = 0.0363;                 #Riser Diameter
R = D/2;                    #Riser Radius
L = 152.524;                #Riser length

time = vec(readdlm("./Data/t.csv")); 
fs = readdlm("./Data/fs.csv");
vel = vec(readdlm("./Data/vel.csv"));
z = -vec(readdlm("./Data/z.csv"));
x = Matrix(CSV.read("./Data/xCF.csv", DataFrame, header = 0 )); x=x*1e-6;

########## (Statistical) Interpolation  of Faulty Sensors via GPR ###################
# This is Pre-Processing

#Kriging Plot 
#=
t = Int(round(rand(1)[]*length(time)));
sol = krig_faulty(vcat(0, x[t,x[t,:].!=0], 0), vcat(0, z[x[t,:].!=0], L), collect(LinRange(0, L, 250)));
plot(collect(LinRange(0, L, 250)),  lw = 3, sol.y, ribbon = sol.y_variance, fillalpha = 0.4, framestyle = :box, label = "Kriging Interpolation", ylims = (-1.1*maximum(abs.(sol.y)), 1.1*maximum(abs.(sol.y))))
scatter!(vcat(0, z[x[t,:].!=0], L), vcat(0, x[t,x[t,:].!=0], 0),label = "Measurements")
scatter!(z[x[t,:].==0], x[t,x[t,:].==0], label = "Faulty Measurements", ylabel = "ε", xlabel = "z (m)")
=#

z, x = faulty_sensors_GPR(z, x, time);
z, x = mirror_ends(z, x);

len_t, len_x = size(x);
len_z = length(z);

########################## Stochastic Mode Search #########################
# Core Optimization/Learning Work

minmode = 10;                                   #minimum low mode
successes = 0;                                  #number of feasible mode sets found
Nsuccess = 15;                                  #Number of feasible points beyond which search is stopped
hashappened = false;                            #Whether enough feasible points have been found to jump to refinement stage

N_trials = 20000;                               #Number of total training trials
explore_trials = Int(round(0.95*N_trials));     #Number of exploration trials (random search)
refinement_trials = N_trials-explore_trials;    #Number of perturbation trials

Jbest = Inf;                                    #initialized best objective value
trials = 0;                                     #Number of trials (counter), always stays less than N_trials

#Perform the search
while trials < N_trials                 

    #trials: This is always less than or equal to N_trials
    trials += 1;

    #Jump to refinemet stage if many feasible points have been found
    if (trials < explore_trials && successes == Nsuccess && ~hashappened)
        trials = N_trials - refinement_trials;          #Jump to refinement stage
        hashappened = true;                             #Enough feasible points were found
    end

    ############################## Calculation of S at current iteration ################################

    global not_feasible = false;            #feasibility of current mode set, initialized to feasible
    Jval = 0;                               #initialized value of objective of current mode estimate to zero

    if trials <= explore_trials #if in exploration stage
        #Generate 35-45 unique sorted random modes in range minmode:minmode+50
        Nm = sample(minmode:minmode+50,rand(35:45), ordered = true, replace = false) 
        global len_Nm = length(Nm);

    else #if in refinement stage 

        Nm = copy(BestModes);

        #Half of the trials are perturbations
        if (isodd(trials)) 
            #Length preserving perturbation
            Nm = perturbe_Nm(copy(Nm),length(Nm))

        #Half of the trials are mode additions or reductions
        else 
            #Add or remove a few modes
            Nm = increase_decrease_Nm(copy(Nm), length(Nm))
        end
        #If you fail to update (due to randomness of perturbations) then try again
        if Nm == BestModes
            trials = trials - 1;
            print("\nFailed to update - Trying new perturbation")
            continue;
        end

        global len_Nm = length(Nm);

    end

    print("\nTrials: ", trials, "\nModes: ", Nm, "\n")

    A = make_A_ls_in_parallel(Nm, len_Nm, z, len_z);
    
    for t = 1:len_t

        #Solve for (optimal in least squares sense) coefficients c, given mode set Nm at time t, i.e. find a_n and b_n
        b = make_b_ls_in_parallel(t, len_Nm, len_z, x)
        c = A\b;
        offset = real(-sum(c));
        
        #Calculate y(z, t_0) and y''(z,t_0) given the coefficients
        global y, ypp, not_feasible = get_y_ypp_in_parallel(Nm, c, offset, z, len_z)

        if not_feasible
            print("Trial ", trials, ": Nonsensible Amplitude \n")
            break
        end

        #Evaluate the objective
        kappa = 1e2;
        lambda = 1e12;
        mu = 1e1;

        #Add Regularization penalities once
        if t == 1; 
            Regu = mu*sum(Nm[Nm.>40]);
            md = kappa*len_Nm;
            Jval += md + Regu;
        end

        rmse = lambda*trapz(z/L, (x[t,:]-ypp*R).^2);
        Jval += rmse;

        #If Jval has exceeded Jbest the set S is suboptimal 
        if (Jval > Jbest)
            print("Trial ", trials, ": is suboptimal \n")
            not_feasible = true;
            break
        end

    end
 
    if not_feasible
        continue
    end

    if Jval < Jbest
        print("New optimal Set S found with J(s) = ", Jval, " \n")
        successes += 1;
        Jbest = Jval;
        global BestModes = copy(Nm);
    end

end

print("Stochastic Mode Search Completed \n")
print("Best Modes: ", BestModes ,"\n")

#Training on 16/20 seconds
BestModes = vec([19, 21, 22, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 40, 41, 42, 44, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 66, 68, 70]);
#Actual Best
BestModes = vec([19, 21, 23, 24, 25, 26, 27, 28, 29, 32, 33, 35, 36, 38, 40, 42, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 66, 67]);

N_interp = 400;
y_sol, ypp_sol, zeval, i1, i2 = make_plotable(BestModes, z, x, N_interp);
zplot = z[i1:i2];
xplot = x[:,i1:i2];

predicted_strain_rms = rms(ypp_sol*R);
given_strain_rms = rms(xplot);


plot(zplot/L, given_strain_rms, seriestype = :scatter,label = "RMS Measurements")
plot!(zeval/L, predicted_strain_rms, lw = 2, xlabel = "L* = z/L", ylabel = "ε", label = "Model Prediction")

t = Int(round(rand(1)[]*length(time)))
plot(zplot/L, xplot[t,:], label = "Measurements", seriestype = :scatter)
plot!(zeval/L, ypp_sol[t,:]*R,label = "Model Prediction", lw=2, ylabel = "ε", xlabel = "L* = z/L", title = "t = 68.54 s")
plot(zeval/L, y_sol[t,:]/D, lw=2, ylabel = "y* = y/D", xlabel = "L* = z/L", label = "Predicted Motion")


plot(zeval/L, rms(y_sol)/D, lw=2,ylabel = "RMS(y/D)", xlabel = "L* = z/L",label = "Predicted Motion")

#heatmap(zeval/L,time[1:200],y_sol[1:200,:]/D)

anim = @animate for t = 1:length(time)
    #p1 = plot(zeval/L, y_sol[t,:]/D, ylims = (-2,2), label = false, lw = 2);
    #p2 = plot(zeval/L, ypp_sol[t,:]*R, lw = 2,label = "Model Prediction");
    #p3 = plot!(zplot, xplot[t,:], seriestype = :scatter, label = "Measurements")
    plot(zeval/L, y_sol[t,:]/D, ylims = (-2,2), label = false, lw = 2, ylabel = "y* = y/D", xlabel = "L* = z/L")
end

gif(anim, fps = 10)

## Save variables

writedlm("y_sol.txt",y_sol);
writedlm("ypp_sol.txt", ypp_sol);
writedlm("z_plot.txt", zplot);
writedlm("x_plot.txt", xplot);
writedlm("zeval.txt", zeval);



########################### Helper Functions #####################

############    Pre - Processing

function make_krig_domain(x)

    V = [zeros(1) for _ in 1:length(x)];

    for i = 1:length(x)
        V[i] = [x[i]];
    end
    
    return PointSet(V)

end

function krig_faulty(vals, x, z)

    data = georef((y=vals,), reshape(x, 1, length(x)))
    domain = make_krig_domain(z)
    problem = EstimationProblem(data, domain, :y)
    solver = Kriging(:y => (variogram=GaussianVariogram(range = 6.5), mean = 0))
    solution = KrigingEstimators.solve(problem, solver)

    return solution

end

function faulty_sensors_GPR(z,x,time)
    
    condit = x[250,:].!=0;
    ztemp = z[condit];

    push!(ztemp,L);
    pushfirst!(ztemp,0);

    for t =  1:length(time)

        xtemp = copy(x[t,condit])

        push!(xtemp, 0);
        pushfirst!(xtemp, 0);

        sol = krig_faulty(xtemp, ztemp, z)

        x[t,:] = sol.y;

        #plot(ztemp, xtemp, seriestype=:scatter)
        #plot!(zq,yq, seriestype=:scatter)
        #plot!(vcat(0,z,L), vcat(0, xtemp2, 0), seriestype=:scatter)

    end

    #Add BC values at z and x
    pushfirst!(z,0);
    push!(z,L);
    x = cat(zeros(1010), x, zeros(1010), dims = 2);

    return z, x

end

function mirror_ends(z, x)
    
    mir = 15;
    lft = -reverse(z[2:mir+1]);
    rgt = L.+reverse(L.-z[end-mir:end-1]);

    lft_str = -x[:,mir+1:-1:2];
    rgt_str = -x[:,end-1:-1:end-mir];

    z = vcat(lft, z, rgt);
    x = cat(lft_str, x, rgt_str, dims = 2);

    return z,x

end
#plot!(z, x[297,:], seriestype=:scatter)


###########  In-Loop Processing

function perturbe_Nm(Nm, len_Nm)

    num_of_perts = rand(1:min(7,len_Nm))
    pert_locs = shuffle(vcat(zeros(Int64, len_Nm-num_of_perts), ones(Int64, num_of_perts))) .== 1

    for i = 1:len_Nm
        if pert_locs[i] == true
            candidate = Nm[i]+rand(-3:3)
            if ~(candidate in Nm)
                Nm[i] = candidate
            end
        end
    end

    return sort(Nm)

end

#Nm = sample(minmode:minmode+50,rand(25:45), ordered = true, replace = false)
#Nm = perturbe_Nm(Nm,length(Nm))

function increase_decrease_Nm(Nm, len_Nm)

    num_of_increase = rand(1:6)

    if bitrand(1) == trues(1)

        #print("Adding ", num_of_increase, " modes \n")

        full_set = Nm[1]:Nm[end];
        missing_from_Nm = indexin(full_set,Nm);
        to_be_included = full_set[findall(missing_from_Nm .== nothing)];
        len_missing_from_Nm = length(to_be_included);

        if len_missing_from_Nm < num_of_increase
            num_of_increase = len_missing_from_Nm;
        end

        new_modes = sample(to_be_included,num_of_increase, ordered = true, replace = false)
        Nm = sort(unique(vcat(Nm,new_modes)))

    else

        #print("Removing ", num_of_increase," modes \n")
        rem_locs = shuffle(vcat(falses(len_Nm-num_of_increase),trues(num_of_increase)));
        deleteat!(Nm,rem_locs);

    end

    return sort(unique(Nm));
    
end

#Nm = sample(minmode:minmode+50,rand(25:45), ordered = true, replace = false)
#Nm = increase_decrease_Nm(Nm,length(Nm))

function make_A_ls(Nm, len_Nm, z, len_z)

    #A: Complex Matrix for OLS problem A'*A = A'*b 
    A = zeros(Complex{Float64}, len_z+1, 2*len_Nm);
    #a: mode conversion from real a_n, b_n to Complex c_n
    a = vcat(-reverse(Nm), Nm);

    Arowi = zeros(Complex{Float64}, 1,2*len_Nm);              #Row of Matrix A
    lastrow = zeros(Complex{Float64}, 1,2*len_Nm);            #One last row for y'''(L) = 0 boundary condition

    for row = 1:len_z
        for n = 1:2*len_Nm

            Arowi[n] = (a[n])^2*exp(im*a[n]*pi/L*z[row]);
            if row == len_z
                lastrow[n] = -im*(a[n]*pi/L)^3*exp(im*a[n]*pi/L*z[row]);
            end

        end

        A[row,:] = Arowi;

    end

    A[end,:] = lastrow;
    return A
end


function make_b_ls(t, len_Nm, len_z, x)

    b = zeros(len_z+1);         #RHS of system of equations for OLS

    for row = 1:len_z
        b[row] = -L^2*x[t,row]./(pi.^2*R);
    end

    return b

end

function get_y_ypp(Nm, c, offset, z, len_z)

    ypp = zeros(Complex{Float64}, len_z);                     #y''(z, t_0)
    y = zeros(Complex{Float64}, len_z);                       #y(z,t_0)

    a = vcat(-reverse(Nm), Nm);
    zeval = z;
    for row = 1:len_z

        sumypp = 0;
        sumy = 0;

        for n = 1:2*length(Nm)

            sumypp += -(pi/L)^2*a[n]^2*c[n]*exp(im*a[n]*pi/L*zeval[row]);
            sumy += c[n]*exp(im*a[n]*pi/L*zeval[row]);           

        end

        if ( abs(real(sumy + offset)/D) > 2 )

            return nothing, nothing, true

        end

        ypp[row] = sumypp;
        y[row] = sumy + offset;

    end

    return real(y), real(ypp), false

end

#y, ypp, not_feasible = get_y_ypp(BestModes, c, offset, z, len_z)

############## In - plotting

function get_full_y_ypp(Nm, c, offset, z, N_interp)

    ypp = zeros(Complex{Float64}, N_interp);                     #y''(z, t_0)
    y = zeros(Complex{Float64}, N_interp);                       #y(z,t_0)

    a = vcat(-reverse(Nm), Nm);
    
    zeval = collect(LinRange(0,L, N_interp))
    for row = 1:length(zeval)

        sumypp = 0;
        sumy = 0;

        for n = 1:2*length(Nm)

            sumypp += -(pi/L)^2*a[n]^2*c[n]*exp(im*a[n]*pi/L*zeval[row]);
            sumy += c[n]*exp(im*a[n]*pi/L*zeval[row]);    
            
        end

        ypp[row] = sumypp;
        y[row] = sumy + offset;

    end

    return real(y), real(ypp), zeval

end

function rms(x)

    if typeof(x) == Vector{Float64}
        x = reshape(x,length(x),1)
    end

    len_t, len_x = size(x);
    out = zeros(len_x)

    for z = 1:len_x
        s = 0;
        for t = 1:len_t
            s += x[t,z]^2
        end
        out[z] = sqrt(1/len_t*s)
    end

    return out;

end

function make_plotable(BestModes, z, x, N_interp = 350)

    Nm = copy(BestModes);
    len_Nm = length(BestModes);

    y_sol = zeros(len_t, N_interp);
    ypp_sol = zeros(len_t, N_interp);

    A = make_A_ls_in_parallel(Nm, len_Nm, z, len_z);
        
    for t = 1:len_t

        b = make_b_ls_in_parallel(t, len_Nm, len_z, x)
        c = A\b;
        offset = -real(sum(c));
        global y, ypp, zeval = get_full_y_ypp_in_parallel(Nm, c, offset, z, N_interp)
        y_sol[t,:] = y;
        ypp_sol[t,:] = ypp; 

    end

    i1 = findfirst(z.>=0);
    i2 = findfirst(z.>=L);

    return y_sol, ypp_sol, zeval, i1, i2

end


############### Helpers in Parallel (with associated tests) ##################
# Currently using 4 threads (8 required too many allocations)
using BenchmarkTools

#Speeds up by ~12.5% 
function make_A_ls_in_parallel(Nm, len_Nm, z, len_z)

    #A: Complex Matrix for OLS problem A'*A = A'*b 
    A = zeros(Complex{Float64}, len_z+1, 2*len_Nm);
    #a: mode conversion from real a_n, b_n to Complex c_n
    a = vcat(-reverse(Nm), Nm);

    Arowi = zeros(Complex{Float64}, 1,2*len_Nm);              #Row of Matrix A
    lastrow = zeros(Complex{Float64}, 1,2*len_Nm);            #One last row for y'''(L) = 0 boundary condition

    for row = 1:len_z
        Threads.@threads for n = 1:2*len_Nm
            Arowi[n] = (a[n])^2*exp(im*a[n]*pi/L*z[row]);
            if row == len_z
                lastrow[n] = -im*(a[n]*pi/L)^3*exp(im*a[n]*pi/L*z[row]);
            end
        end
        A[row,:] = Arowi;
    end

    A[end,:] = lastrow;
    return A
end

Areg= @btime make_A_ls(Nm, length(Nm), z, length(z));              #About ~800 μs
Apar = @btime make_A_ls_in_parallel(Nm, length(Nm), z, length(z));    #About ~700 μs
Areg == Apar

Areg= @belapsed make_A_ls(Nm, length(Nm), z, length(z))             
Apar = @belapsed make_A_ls_in_parallel(Nm, length(Nm), z, length(z))      


#Speeds up by ~60%
function make_b_ls_in_parallel(t, len_Nm, len_z, x)

    b = zeros(len_z+1);         #RHS of system of equations for OLS

    Threads.@threads for row = 1:len_z
        b[row] = -L^2*x[t,row]./(pi.^2*R);
    end

    return b

end

breg = @belapsed make_b_ls(560, length(Nm), length(z), x)        
bpar = @belapsed make_b_ls_in_parallel(560, length(Nm), length(z), x)     
breg == bpar;

function get_y_ypp_in_parallel(Nm, c, offset, z, len_z)

    ypp = [Threads.Atomic{Float64}(0.0) for _ in 1:len_z];
    y = [Threads.Atomic{Float64}(0.0) for _ in 1:len_z];

    a = vcat(-reverse(Nm), Nm);
    
    zeval = z;
    for row = 1:len_z

        Threads.@threads for n = 1:2*length(Nm)
       
            Threads.atomic_add!(ypp[row], real(-(pi/L)^2*a[n]^2*c[n]*exp(im*a[n]*pi/L*zeval[row])))
            Threads.atomic_add!(y[row], real(c[n]*exp(im*a[n]*pi/L*zeval[row])))

        end

        if ( abs(y[row][] + offset)/D) > 2 
            return nothing, nothing , true
        end

    end

    return [(y[i][]+offset) for i in 1:len_z], [ypp[i][] for i in 1:len_z], false

end

A = make_A_ls(Nm, length(Nm), z, length(z));
t = Int(round(rand(1)[]*len_t))
b = make_b_ls(t, length(Nm), length(z), x);
c = A\b;
offset = real(-sum(c));

yreg, yppreg, not_feasible = @btime get_y_ypp(Nm, c, offset, z, length(z));
ypar, ypppar, not_feasiblepar = @btime get_y_ypp_in_parallel(Nm,c,offset,z,length(z));

t1 = @belapsed get_y_ypp(Nm, c, offset, z, length(z))
t2 = @belapsed get_y_ypp_in_parallel(Nm,c,offset,z,length(z))

maximum(abs.(yreg)-abs.(ypar))
maximum(abs.(yppreg)-abs.(ypppar))
not_feasible == not_feasiblepar

function get_full_y_ypp_in_parallel(Nm, c, offset, z, N_interp)

    ypp = [Threads.Atomic{Float64}(0.0) for _ in 1:N_interp];
    y = [Threads.Atomic{Float64}(0.0) for _ in 1:N_interp];

    a = vcat(-reverse(Nm), Nm);
    zeval = collect(LinRange(0,L, N_interp))

    Threads.@threads for row = 1:length(zeval)

        Threads.@threads for n = 1:2*length(Nm)
       
            Threads.atomic_add!(ypp[row], real(-(pi/L)^2*a[n]^2*c[n]*exp(im*a[n]*pi/L*zeval[row])))
            Threads.atomic_add!(y[row], real(c[n]*exp(im*a[n]*pi/L*zeval[row])))

        end

    end

    return [(y[i][]+offset) for i in 1:length(zeval)], [ypp[i][] for i in 1:length(zeval)], zeval

end

A = make_A_ls(Nm, length(Nm), z, length(z));
t = Int(round(rand(1)[]*len_t))
b = make_b_ls(t, length(Nm), length(z), x);
c = A\b;
offset = real(-sum(c));
N_interp = 450;

yreg, yppreg, zevalreg = @btime get_full_y_ypp(Nm, c, offset, z, N_interp);
ypar, ypppar, zevalpar = @btime get_full_y_ypp_in_parallel(Nm,c,offset,z,N_interp);

maximum(abs.(yreg)-abs.(ypar))
maximum(abs.(yppreg)-abs.(ypppar))
zevalreg == zevalpar



############### Old less efficient helpers ################

#=
function interp_faulty_sensors(z,x,time)
    
    condit = x[250,:].!=0;
    ztemp = z[condit];

    push!(ztemp,L);
    pushfirst!(ztemp,0);

    for t =  1:length(time)

        xtemp = copy(x[t,condit])

        push!(xtemp, 0);
        pushfirst!(xtemp, 0);

        spline = CubicSpline(ztemp, xtemp);

        #zq = range(ztemp[1], stop = ztemp[end], length = 72)
        zq = z[condit.==0]
        yq = spline[zq]

        xtemp2 = copy(x[t,:])
        #push!(xtemp,0); pushfirst!(xtemp, 0);
        xtemp2[condit.==0] = yq;

        x[t,:] = xtemp2;

        #plot(ztemp, xtemp, seriestype=:scatter)
        #plot!(zq,yq, seriestype=:scatter)
        #plot!(vcat(0,z,L), vcat(0, xtemp2, 0), seriestype=:scatter)

    end

    #Add BC values at z and x
    pushfirst!(z,0);
    push!(z,L);

    x = cat(zeros(1010), x, zeros(1010), dims = 2);

    return z, x

end

function make_krig_domain(x_start, x_end, Npoints)

    V = [zeros(1) for _ in 1:Npoints]
    x = collect(LinRange(x_start, x_end, Npoints))

    for i = 1:length(x)
        V[i] = [x[i]]
    end

    return PointSet(V)

end

function krig_faulty(vals, x, gridstart, gridend, Npoints)
    #y is a vector
    #x is a vector
    #geodata = georef((y=[1.23,5.35,8.64,9.23,7.76],), [1.0 2.0 3.0 4.0 5.0])
    geodata = georef((y=vals,), reshape(x, 1, length(x)))
    geodomain = make_krig_domain(gridstart, gridend, Npoints)
    problem = EstimationProblem(geodata, geodomain, :y)
    solver = Kriging(:y => (variogram=GaussianVariogram(range = 6.5), mean = 0))
    solution = KrigingEstimators.solve(problem, solver)

    return solution

end

function make_least_squares_system(t, Nm, len_Nm, z, len_z, x)

    #A: Complex Matrix for OLS problem A'*A = A'*b 
    A = zeros(Complex{Float64}, len_z+1, 2*len_Nm);
    #a: mode conversion from real a_n, b_n to Complex c_n
    a = vcat(-reverse(Nm), Nm);

    Arowi = zeros(Complex{Float64}, 1,2*len_Nm);              #Row of Matrix A
    lastrow = zeros(Complex{Float64}, 1,2*len_Nm);            #One last row for y'''(L) = 0 boundary condition
    b = zeros(len_z+1);                                       #RHS of system of equations for OLS


    for row = 1:len_z
        for n = 1:2*length(Nm)

            Arowi[n] = (a[n])^2*exp(im*a[n]*pi/L*z[row]);
            if row == len_z
                lastrow[n] = -im*(a[n]*pi/L).^3*exp(im*a[n]*pi/L*z[row]);
            end

            A[row,:] = Arowi;
            b[row] = -L^2*x[t,row]./(pi.^2*R);

        end
    end


    A[end,:] = lastrow;

    if rank(A) < length(A[1,:])
        print("trial ", trials, " : Rank Deficient \n")
        return A, b, true
    end

    return A, b, false

end
=#