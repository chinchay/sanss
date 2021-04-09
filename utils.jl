# using TimerOutputs # for `to`
## Create a TimerOutput, this is the main type that keeps track of everything.
# const to = TimerOutput()

# using DelimitedFiles # for writedlm()

include("./customreadlammps.jl")
using .readLammpsModule
# using PyPlot
# using Plots
using DelimitedFiles
#
using StatsBase
using Statistics # for `std`
### \notit for `not in` : https://stackoverflow.com/questions/59978282/is-there-an-elegant-way-to-do-not-in-in-julia

using Random # for `randperm`
using Test # for @test



################################################################################
# DETERMINE Tmin AND Tmax
function getTfromSamples(   samples_r::Array{Array{Array{Int64,1},1}},
                            UionAionBlist::Array{Float64,4}
                        )
    n = length(samples_r)
    @test n >= 3 # we need 3 samples at list to calculate dEmin and dEmax
    list = zeros(n)
    for (i, sample) in enumerate(samples_r)
        Ut   = getEnergy_ion_atoms(sample, U)
        Uion = getEnergy_ion_ion(sample, UionAionBlist)
        e0   = energyBase + Ut + Uion
        list[i] = e0
    end
    deltas = zeros(n - 1)
    for i in 1:n-1
        # println(list[i+1], " ... ", list[i])
        deltas[i] = abs(list[i+1] - list[i])
    end
    k = 1.0
    Tmax = maximum(deltas) * k
    Tmin = minimum(deltas) * k
    return Tmin, Tmax
end

function plotTmaxTminFromSamples(
                L1::Int64,
                L::Int64,
                Nv::Int64,
                ion1::Int,
                ion2::Int,
                removedSites::Array{Int,1},
                U::Array{Float64, 2},
                UionAionBlist::Array{Float64,4},
                L_list::Array{Int,1},
                Ne_list::Array{Int,1}
                )
    #
    x = []
    lTmin = []
    lTmax = []
    # listSamples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    #                 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    #                 5000, 10000, 20000, 40000, 100000, 200000]
    listSamples = [10, 20, 30, 40, 50]
    #
    for nSamples in listSamples
        # nSamples = 70
        samples_r = [ [ zeros(Int64, 4) for i in 1:L ] for w in 1:nSamples]
        samples_r[1] = get_refill(L1, L, Nv, ion1, ion2, removedSites)
        for i in 2:nSamples
            notRefill = get_notRefill(samples_r[i-1], Nv)
            dE, r_new, n_new = get_dE2(L, samples_r[i-1], notRefill,
                            removedSites, U, UionAionBlist, L_list, Ne_list)
            samples_r[i] = deepcopy(r_new)
            notRefill = copy(n_new)
        end
        Tmin, Tmax = getTfromSamples( samples_r, UionAionBlist )
        append!(x, nSamples)
        append!(lTmin, Tmin)
        append!(lTmax, Tmax)
        println(Tmin, " ", Tmax)
    end
    #
    # using PyPlot
    PyPlot.close() # it gave me errors saying that other plots were open.
    # PyPlot.close(fig)
    fig, ax1 = PyPlot.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, lTmax, "--r", marker="o", markersize=4.5)
    ax2.plot(x, lTmin, "--b", marker="o", markersize=4.5)
    ax1.set_xlabel("length of random walk")
    ax1.set_ylabel("Tmax (a.u.)", color="r")
    ax2.set_ylabel("Tmin (a.u.)", color="b")
    ax1.tick_params(axis="y", colors="red")
    ax2.tick_params(axis="y", colors="blue")
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    # ax1.set_ylim([-5600, -5150])
    # PyPlot.show() # <<-- this won't work. https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so
    PyPlot.savefig("mygraph.png")
    PyPlot.close(fig)
    ##
    ### Tmax ~ 350 and Tmin ~ 0
end
################################################################################

# getInitWalkers2( maximumMoves, refill, notRefill, L, UionAionBlist, nWalkers, energyBase)
function getInitWalkers2(
                    tempLength::Int64,
                    maximumMoves::Int64,
                    refill::Array{Int64,2},
                    notRefill::Array{Int64,1},
                    L::Int64,
                    UionAionBlist::Array{Float64,4},
                    nWalkers::Int64,
                    energyBase::Float64
                    )
    # w_r     = [ [ zeros(Int64, 4) for i in 1:L ] for w in 1:nWalkers]
    w_r     = zeros(Int64, 4, L, nWalkers) # 4×L×nWalkers ::Array{Float64,3}:

    # w_n     = [ zeros(Int64, Ne) for w in 1:nWalkers]
    w_n     = zeros(Int64, Ne, nWalkers)

    # w_r_opt = [ [ zeros(Int64, 4) for i in 1:L ] for w in 1:nWalkers]
    w_r_opt = zeros(Int64, 4, L, nWalkers)

    # w_n_opt = [ zeros(Int64, Ne) for w in 1:nWalkers]
    w_n_opt = zeros(Int64, Ne, nWalkers)

    w_e0         = zeros(nWalkers)
    w_dEsum      = zeros(nWalkers)
    w_dEsum_opt  = zeros(nWalkers)
    # w_E   = [ zeros(0) for w in 1:nWalkers]
    w_E   = zeros(maximumMoves, nWalkers) # 0×1 Array{Float64,2}
    # w_acc = [ zeros(Bool, 0) for w in 1:nWalkers]
    w_acc = zeros(Bool, maximumMoves, nWalkers) # maximumMoves×2 Array{Bool,2}
    record_T  = zeros(maximumMoves)
    listT     = zeros(tempLength)
    w_T_opt   = -ones(nWalkers) ## zeros(nWalkers) I changed from zeros to -ones to check if it has changed in running
    isOptimal = zeros(Bool, nWalkers)
    #
    Ut   = getEnergy_ion_atoms(L, refill, U)
    Uion = getEnergy_ion_ion( L, refill, UionAionBlist)
    e0   = energyBase + Ut + Uion
    for w in 1:nWalkers
        # w_r[w]     = deepcopy(refill)
        # w_n[w]     = copy(notRefill)
        # w_r_opt[w] = deepcopy(w_r[w])
        # w_n_opt[w] = copy(w_n[w])
        # w_e0[w]    = e0

        for i in 1:L
            for j in 1:4
                w_r[j, i, w]     = refill[j, i]
                w_r_opt[j, i, w] = refill[j, i]
            end
        end
        for i in 1:Ne
            w_n[i, w]     = notRefill[i]
            w_n_opt[i, w] =  notRefill[i]
        end
        #
        w_e0[w]    = e0

        # w_dEsum[w] = 0.0
        # w_dEsum_opt[w]  = 0.0
    end
    # walker_dEsum = zeros(nWalkers) # <<-- THIS WON'T WOKR IN JULIA!! a new local variable will be generated :(
    # w_dEsum_opt  = zeros(nWalkers) # <<-- THIS WON'T WOKR IN JULIA!! a new local variable will be generated :(
    #
    return w_r, w_n, w_r_opt, w_n_opt, w_e0, w_dEsum, w_dEsum_opt, w_E, w_acc, record_T, listT, w_T_opt, isOptimal
end

# getBestFromScanNeighs!(L, r, n, p_dE_best, removedSites, U, UionAionBlist, L_list, rTemp, nTemp, rTemp2, nTemp2)
function getBestFromScanNeighs!(
        L::Int64,
        r::Array{Int64,2},
        n::Array{Int64,1},
        p_dE_best::Array{Float64,1},
        removedSites::Array{Int,1},
        U::Array{Float64, 2},
        UionAionBlist::Array{Float64,4},
        L_list::Array{Int,1},
        rTemp::Array{Int64,2},
        nTemp::Array{Int64,1},
        rTemp2::Array{Int64,2},
        nTemp2::Array{Int64,1},
        nCfgNeighbors::Int
        )
    #
    # nCfgNeighbors = 1 # @@@@@@@@@@@@@@@@@@@
    # dE_best = 0.0
    # myCopyRefill!(L, r, rTemp)
    # myCopyNotRefill!(Ne, n, nTemp)

    # this is the first neighbor:
    p_dE_best[1] = 0.0
    get_dE2!(L, Ne, r, n, removedSites, U, UionAionBlist, L_list, rTemp, nTemp, p_dE_best)
    dE = p_dE_best[1]
    #
    dE_best = dE

    # the following neighbors:
    for j in 1:nCfgNeighbors - 1
        p_dE_best[1] = 0.0
        get_dE2!(L, Ne, r, n, removedSites, U, UionAionBlist, L_list, rTemp2, nTemp2, p_dE_best)
        dE = p_dE_best[1]
        #
        if dE < dE_best
            dE_best = dE
            myCopyRefill!(L, rTemp2, rTemp)
            myCopyNotRefill!(Ne, nTemp2, nTemp)
        end
    end

    p_dE_best[1] = dE_best
    # return r_best, n_best, dE_best
end

# move_equilibration2!( L, Ne, r, n, p_accepted, p_dEsum, p_dE_best, removedSites, U, UionAionBlist, L_list, T, rTemp, nTemp, rTemp2, nTemp2 )
function move_equilibration2!(
    L::Int64,
    Ne::Int,
    r::Array{Int64,2},
    n::Array{Int64,1},
    p_accepted::Array{Bool,1},
    p_dEsum::Array{Float64,1},
    p_dE_best::Array{Float64,1},
    removedSites::Array{Int,1},
    U::Array{Float64, 2},
    UionAionBlist::Array{Float64,4},
    L_list::Array{Int,1},
    T::Float64,
    rTemp::Array{Int64,2},
    nTemp::Array{Int64,1},
    rTemp2::Array{Int64,2},
    nTemp2::Array{Int64,1},
    nCfgNeighbors::Int,
    Eold::Float64
    )
    #
    # dE, v, ib, B, ionA, w, ia = get_dE1(r, n, removedSites, U, UionAionBlist, L_list, Ne_list)

    # choosing the best neighbor
    # @timeit to "a" r_best, n_best, dE_best = getBestFromScanNeighs( L, r, n, removedSites, U, UionAionBlist, L_list, Ne_list )

    ############################################################
    p_dE_best[1] = 0.0
    getBestFromScanNeighs!(L, r, n, p_dE_best, removedSites, U, UionAionBlist, L_list, rTemp, nTemp, rTemp2, nTemp2, nCfgNeighbors)
    dE = p_dE_best[1]
    ############################################################

    # @timeit to "b" dE_best, r_best, n_best = get_dE2(L, r, n, removedSites, U, UionAionBlist, L_list, Ne_list)
    # Ne = length(n)
    # @timeit to "**"
    # sumdUdistr = 0.0
    # @timeit to "a1" dE_best, r_best, n_best = get_dE2(L, Ne, r, n, removedSites, U, UionAionBlist, L_list, rTemp, nTemp)

    ############################################################
    # p_dE_best[1] = 0.0
    # get_dE2!(L, Ne, r, n, removedSites, U, UionAionBlist, L_list, rTemp, nTemp, p_dE_best)
    # dE = p_dE_best[1]
    ############################################################

    # apply the best neighbor to the metropolis test

    accepted = metropolis(dE, T)
    # phi=0.5
    # accepted = metropolisBounded(dE, T, Eold, phi)

    # accepted = false
    # if dE <= 0.0
    # accepted = true
    # elseif getBoltzmanFactor(dE, T) > rand()
    # accepted = true
    # end
    #
    # println(accepted)
    if accepted
        # println("passed")
        #swap
        # @timeit to "b2" r = r_best
        # @timeit to "b3" n = n_best
        # myCopyRefill!(rTemp, r)
        myCopyRefill!(L, rTemp, r)

        # myCopyNotRefill!(nTemp, n)
        myCopyNotRefill!(Ne, nTemp, n)

        # dEsum += dE
        p_dEsum[1] += dE
    end
    #
    # return r, n, accepted, dEsum, dE_best

    p_accepted[1] = accepted
    # p_dEsum[1] = dEsum ## content of `p_dE_best` already mutated
    # p_dE_best[1] = dE_best ## content of `p_dE_best` already mutated
    #  also mutating the content of r and n


    # return accepted, dEsum, dE_best
end

# dE_best, v_best, ib_best, B_best, ionA_best, w_best, ia_best = getBestFromScanNeighSimulation(L, Ne, r, n, removedSites, U, UionAionBlist)
function getBestFromScanNeighSimulation(
            L::Int,
            Ne::Int,
            r::Array{Int64,2},
            n::Array{Int64,1},
            removedSites::Array{Int,1},
            U::Array{Float64, 2},
            UionAionBlist::Array{Float64,4},
            nCfgNeighbors::Int
    )

    # nCfgNeighbors = 1

    # this is the first neighbor:
    dE, v, ib, B, ionA, w, ia = get_dE1(L, Ne, r, n, removedSites, U, UionAionBlist)
    #
    dE_best = dE
    v_best  = v
    ib_best = ib
    B_best  = B
    ionA_best = ionA
    w_best  = w
    ia_best = ia

    # the following neighbors:
    for j in 1:nCfgNeighbors - 1
        dE, v, ib, B, ionA, w, ia = get_dE1(L, Ne, r, n, removedSites, U, UionAionBlist)        #
        if dE < dE_best
            dE_best = dE
            v_best  = v
            ib_best = ib
            B_best  = B
            ionA_best = ionA
            w_best  = w
            ia_best = ia
        end
    end
    #
    return dE_best, v_best, ib_best, B_best, ionA_best, w_best, ia_best
end

function metropolis(dE::Float64, T::Float64)
    accepted = false
    if dE <= 0.0
        accepted = true
    elseif getBoltzmanFactor(dE, T) > rand()
        accepted = true
    end
    return accepted
end

function metropolisBounded(dE::Float64, T::Float64, Eold::Float64, phi::Float64)
    accepted = false
    if dE <= 0.0
        accepted = true
    elseif dE <= Eold * ( phi - 1.0) # factorBM
        if getBoltzmanFactor(dE, T) > rand()
            accepted = true
        end
    end
    return accepted
end

# move_simulation!( L, Ne, r, n, p_accepted, p_dEsum, p_dE_best, removedSites, U, UionAionBlist, T)
function move_simulation!(
                        L::Int,
                        Ne::Int,
                        r::Array{Int64,2},
                        n::Array{Int64,1},
                        p_accepted::Array{Bool,1},
                        p_dEsum::Array{Float64,1},
                        p_dE_best::Array{Float64,1},
                        removedSites::Array{Int,1},
                        U::Array{Float64, 2},
                        UionAionBlist::Array{Float64,4},
                        T::Float64,
                        nCfgNeighbors::Int,
                        Eold::Float64
                        )
    #
    ############################################################
    # dE, v, ib, B, ionA, w, ia = get_dE1(L, Ne, r, n, removedSites, U, UionAionBlist)
    #
    dE, v, ib, B, ionA, w, ia = getBestFromScanNeighSimulation(L, Ne, r, n, removedSites, U, UionAionBlist, nCfgNeighbors)
    p_dE_best[1] = dE
    ############################################################
    #
    # accepted = false
    # if dE <= 0
    #     accepted = true
    # elseif getBoltzmanFactor(dE, T) > rand()
    #     accepted = true
    # end
    #
    accepted = metropolis(dE, T)
    # phi=0.95
    # accepted = metropolisBounded(dE, T, Eold, phi)
    #
    if accepted
        # println("passed")
        #swap
        # r[v] = [ v, ib, B, ionA ] # this assignment will take TOO, EXCESIVE, MUCH TIME!!, better to do:
        r[1, v] = v
        r[2, v] = ib
        r[3, v] = B
        r[4, v] = ionA
        #
        n[w] = ia
        p_dEsum[1] += dE
    end
    #
    p_accepted[1] = accepted
    # return r, n, accepted, dEsum, dE
    # return Any[r, n, accepted, dEsum, dE]
end

# getListT!( To, tempLength, scheme, listT)
function getListT!(
        To::Float64,
        tempLength::Int64,
        scheme::String,
        listT::Array{Float64,1}
        )
    # listT = zeros(tempLength)
    if scheme == "linear"
        Tf = To / 10
        # Tf = To / 2.0
        dT = (Tf - To) / tempLength
        for i in 1:tempLength
            listT[i] =  To + ( (i - 1) * dT)
        end
    elseif scheme == "constant"
        for i in 1:tempLength
            listT[i] = To
        end
    end
    # return listT
end





function getWalkerTotalEnergy(
                                L::Int,
                                energyBase::Float64,
                                r::Array{Int64,2},
                                U::Array{Float64, 2},
                                UionAionBlist::Array{Float64,4}
                            )
	Ut         = getEnergy_ion_atoms(L, r, U)
	Uion       = getEnergy_ion_ion(L, r, UionAionBlist)
	return energyBase + Ut + Uion
end

# returnToPreviousOpt!(L, Ne, nWalkers, keepLooping, w_r, w_n, w_r_opt, w_n_opt, isOptimal, w_dEsum_opt, w_dEsum, w_e0, w_T_opt )
function returnToPreviousOpt!(
    L::Int,
    Ne::Int,
    nWalkers::Int,
    keepLooping::Bool,
    w_r::Array{Int64,3},
    w_n::Array{Int64,2},
    w_r_opt::Array{Int64,3},
    w_n_opt::Array{Int64,2},
    isOptimal::Array{Bool,1},
    w_dEsum_opt::Array{Float64,1},
    w_dEsum::Array{Float64,1},
    w_e0::Array{Float64,1},
    w_T_opt::Array{Float64,1},
    w_E::Array{Float64,2},
    moves::Int
    )
    #
    # return to the previous optimal solution after sptesTconstant loop:
    for w in 1:nWalkers
        if !keepLooping
            # println("w_r_opt will be copied to w_r")
            # println("You should check now if energies of w_r have the following energies:")
        end
        # if w_r[w] != w_r_opt[w]
        if !isOptimal[w]
            # println("oooooooo")
            # w_r[w] = deepcopy(w_r_opt[w])

            # w_r[w] = copy(w_r_opt[w])
            # w_n[w] = copy(w_n_opt[w])

            for i in 1:L::Int
                for j in 1:4
                    w_r[j,i,w] = w_r_opt[j,i,w]
                end
            end
            for i in 1:Ne
                w_n[i,w] = w_n_opt[i,w]
            end

            w_dEsum[w]    = w_dEsum_opt[w]
            w_E[moves, w] = w_e0[w] + w_dEsum[w]
            if !keepLooping
                # println( "energy of opt: ", w_e0[w] + w_dEsum[w] )
                println( "energy of opt: ", w_E[moves, w] )
                println( "found at T = ", w_T_opt[w])
            end
        end
    end
end

function getNeighborsToScan(T::Float64)
    if 100.0 < T 
        return 1
	elseif belongs(T, 30.0, 100.0)
        return 100
    elseif T < 30.0
        return 1
    end
end

# function getNeighborsToScan(T::Float64)
#     if 800.0 < T 
#         return 1
# 	elseif belongs(T, 0.0, 800.0)
#         return 100
#     else
#         return 1
#     end
# end

# function getNeighborsToScan(T::Float64) ## el "grueso"
#     # nCfgNeighbors = 1
#     if 200.0 < T 
#         return 1
#     elseif belongs(T, 100.0, 200.0)
#         return 10
#     elseif belongs(T, 50.0, 100.0)
#         return 20
#     elseif T < 50.0
#         return 40
#     end
# end

function RURS_1(
    refill::Array{Int64,2},
    notRefill::Array{Int64,1},
    L::Int64,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    r::Array{Int64,2},
    n::Array{Int64,1},
    displayMessages::Bool
    # nCfgNeighbors::Int
    )
    #
    if displayMessages
        println("****** SIMULATION STAGE ******")
        println("moves to perform: ", steps)
        println("Tmax:             ", Tmax)
        println("alpha:            ", alpha)
        println("walkersBornEqual: ", walkersBornEqual)
        println("tempLength:       ", tempLength)
        println("scheme:           ", scheme)
        println("nWalkers:         ", nWalkers)
    end
    #
    # _, _, w_r_opt, w_n_opt, w_e0, w_dEsum, w_dEsum_opt, w_E, w_acc =
    #     getInitWalkers(L1, L, Nv, ion1, ion2, removedSites,
    #                     UionAionBlist, nWalkers, walkersBornEqual, energyBase)
    # #
    # # w_r, w_n are given in the input
    # # Correct things:
    # for w in 1:nWalkers
    #     w_r_opt[w] = deepcopy(w_r[w])
    #     w_n_opt[w] = copy(w_n[w])
    #     Ut         = getEnergy_ion_atoms(w_r[w], U)
    #     Uion       = getEnergy_ion_ion(w_r[w], UionAionBlist)
    #     w_e0[w]    = energyBase + Ut + Uion
    # end
    # #

    countMiddle = 0
    maximumMoves = steps * tempLength
    #
    w_r, w_n, w_r_opt, w_n_opt,
        w_e0, w_dEsum, w_dEsum_opt, w_E, w_acc, record_T, listT, w_T_opt, isOptimal =
        getInitWalkers2( tempLength, maximumMoves, refill, notRefill, L, UionAionBlist, nWalkers, energyBase )
    #

    To = Tmax
    # alpha = 0.9
    #
    moves = 0

    lastAccepted = 0 # assuming just 1 walker @@@@@@@@######@@@@@@@@@
    

    # arrays containig just one element, to avoid memory usage
    p_accepted = zeros(Bool,1) # will contain `accepted`
    p_dEsum = zeros(1) # will contain `dEsum`
    p_dE = zeros(1) # will contain `dE`

    lastE = 0
    #
    keepLooping = true
    for _ in 1:steps
        if keepLooping
            getListT!( To, tempLength, scheme, listT)

            # @timeit to "n1b" for j in 1:tempLength
            # @timeit to "n1c" T = listT[j]
            for T in listT
                if keepLooping
                    ####################################################################
                    # move:
                    moves += 1

                    # if moves <= 40_000
                    #
                    for w in 1:nWalkers
                        #
                        matrixToSlice!(w, L, Ne, w_r, w_n, r, n)
                        #
                        p_accepted[1] = false
                        p_dEsum[1]    = w_dEsum[w]
                        p_dE[1]       = 0.0

                        if moves > 1
                            Eold = w_E[moves - 1, w]
                        else
                            Eold = w_e0[w]
                        end
                        #

                        
                        nCfgNeighbors = getNeighborsToScan(T)
                        # nCfgNeighbors = 1
                        # nCfgNeighbors = 100
                        # nCfgNeighbors = 40
                        # nCfgNeighbors = 100

                        # It will mutate p_accepted, p_dEsum, p_dE:
                        move_simulation!( L, Ne, r, n, p_accepted, p_dEsum, p_dE, removedSites, U, UionAionBlist, T, nCfgNeighbors, Eold)
                        #
                        accepted = p_accepted[1]
                        w_dEsum[w] = p_dEsum[1]

                        if accepted
                            sliceToMatrix!(w, L, Ne, r, n, w_r, w_n)
                            lastAccepted = moves
                                #
                            if w_dEsum[w] < w_dEsum_opt[w]
                                w_dEsum_opt[w] = w_dEsum[w]
                                w_T_opt[w] = T # save the temperature in which `opt` was found
                                sliceToMatrix!(w, L, Ne, r, n, w_r_opt, w_n_opt)
                                isOptimal[w] = true
                            else
                                isOptimal[w] = false
                            end
                        else
                            isOptimal[w] = false
                        end
                        #
                        w_acc[moves, w] = accepted
                        w_E[moves, w] = w_e0[w] + w_dEsum[w]
                    end
                    #
                    # end
                    #
                    
                    if moves == 40_000
                        for w in 1:nWalkers
                            if w_e0[w] + w_dEsum_opt[w] < -5590.0
                                countMiddle += 1
                            end
                        end
                    end
                    
                    ####################################################################
                    record_T[moves] = T
                    ####################################################################
                    if T <= 1.0 ## stop program immediately if T reaches Tmin=1
                        keepLooping = false
                        println( "This RURS version stops when temperature reaches 1. It does not wait to complete all moves" )
                    end
                end
            end
            #
            # before updating with the optimal configs and optimal energies, Elena asked to show standard deviation of
            # the energy at the end of simulation for different runs:
            if nWalkers == 1
                lastE = w_e0[1] + w_dEsum[1]
            end
            #
            # return to the previous optimal solution after sptesTconstant loop:
            # keepLooking = true
            returnToPreviousOpt!(L, Ne, nWalkers, keepLooping, w_r, w_n, w_r_opt, w_n_opt, isOptimal, w_dEsum_opt, w_dEsum, w_e0, w_T_opt, w_E, moves )

            #
            # # return to the previous optimal solution after sptesTconstant loop:
            # for i in 1:nWalkers
            #     if walkers_r[i] != w_r_opt[i]
            #         # println("oooooooo")
            #         # walkers_r[i] = copy.deepcopy(w_r_opt[i])
            #         walkers_r[i] = copy(w_r_opt[i])
            #         walkers_n[i] = copy(w_n_opt[i])
            #         walker_dEsum[i] = w_dEsum_opt[i]
            #     end
            # end

            # # # remove the energetic walker, and clone randomly one of the others:
            # if nWalkers > 1
            #     # iMax  = np.argmax(walker_dEsum)
            #     _, iMax = findmax(walker_dEsum)
            #     iRand = rand( [i for i in 1:nWalkers if i != iMax] )
            #     #
            #     walkers_r[iMax] = deepcopy(walkers_r[iRand])
            #     walkers_n[iMax] = copy(walkers_n[iRand])
            #     walker_dEsum[iMax] = walker_dEsum[iRand]
            #     w_e0[iMax] = w_e0[iRand]
            # end
            ####################################################################
            To *= alpha # geometric cooling scheme
        end
    end
    #
    #
    #
    # println("RURS finished.")
    return w_r, w_n, w_E, w_acc, record_T, countMiddle, lastE
end


function RURS_2(
    refill::Array{Int64,2},
    notRefill::Array{Int64,1},
    L::Int64,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    Tmin::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    r::Array{Int64,2},
    n::Array{Int64,1},
    displayMessages::Bool
    # nCfgNeighbors::Int
    )
    #
    if displayMessages
        println("****** SIMULATION STAGE ******")
        println("moves to perform: ", steps)
        println("Tmax:             ", Tmax)
        println("alpha:            ", alpha)
        println("walkersBornEqual: ", walkersBornEqual)
        println("tempLength:       ", tempLength)
        println("scheme:           ", scheme)
        println("nWalkers:         ", nWalkers)
    end
    #
    countMiddle = 0
    maximumMoves = steps * tempLength
    #
    w_r, w_n, w_r_opt, w_n_opt,
        w_e0, w_dEsum, w_dEsum_opt, w_E, w_acc, record_T, listT, w_T_opt, isOptimal =
        getInitWalkers2( tempLength, maximumMoves, refill, notRefill, L, UionAionBlist, nWalkers, energyBase )
    #

    To = Tmax
    # alpha = 0.9
    #
    move = 0

    lastAccepted = 0 # assuming just 1 walker @@@@@@@@######@@@@@@@@@
    

    # arrays containig just one element, to avoid memory usage
    p_accepted = zeros(Bool,1) # will contain `accepted`
    p_dEsum = zeros(1) # will contain `dEsum`
    p_dE = zeros(1) # will contain `dE`

    lastE = 0
    #
    
    totalMoves = steps * tempLength
    temperatureList = zeros(totalMoves)
    getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )

    keepLooping = true
    move = 0

    acceptanceRate = 1 ###@@@@@@@@@@@@@@@@@@@ assumed 1 WALKER!!!! @@@@@@@@
    nAccepted = 0  ###@@@@@@@@@@@@@@@@@@@ assumed 1 WALKER!!!! @@@@@@@@
    listAcceptanceRate = zeros(totalMoves)


    for T in temperatureList
        move += 1
        #
        if keepLooping
            # getListT!( To, tempLength, scheme, listT)

            # @timeit to "n1b" for j in 1:tempLength
            # @timeit to "n1c" T = listT[j]

            # if move <= 40_000
            #
            for w in 1:nWalkers
                #
                matrixToSlice!(w, L, Ne, w_r, w_n, r, n)
                #
                p_accepted[1] = false
                p_dEsum[1]    = w_dEsum[w]
                p_dE[1]       = 0.0

                if move > 1
                    Eold = w_E[move - 1, w]
                else
                    Eold = w_e0[w]
                end
                #

                
                # nCfgNeighbors = getNeighborsToScan(T)
                nCfgNeighbors = 1
                # nCfgNeighbors = 100
                # nCfgNeighbors = 40
                # nCfgNeighbors = 100

                # It will mutate p_accepted, p_dEsum, p_dE:
                move_simulation!( L, Ne, r, n, p_accepted, p_dEsum, p_dE, removedSites, U, UionAionBlist, T, nCfgNeighbors, Eold)
                #
                accepted = p_accepted[1]
                w_dEsum[w] = p_dEsum[1]

                if accepted
                    sliceToMatrix!(w, L, Ne, r, n, w_r, w_n)
                    lastAccepted = move
                    
                    nAccepted += 1
                    #
                    if w_dEsum[w] < w_dEsum_opt[w]
                        w_dEsum_opt[w] = w_dEsum[w]
                        w_T_opt[w] = T # save the temperature in which `opt` was found
                        sliceToMatrix!(w, L, Ne, r, n, w_r_opt, w_n_opt)
                        isOptimal[w] = true
                    else
                        isOptimal[w] = false
                    end
                else
                    isOptimal[w] = false
                end
                #
                w_acc[move, w] = accepted
                w_E[move, w] = w_e0[w] + w_dEsum[w]

                
                if move < 500
                    listAcceptanceRate[move] = nAccepted / move
                elseif move == 500
                    acceptanceRate = nAccepted / move
                    listAcceptanceRate[move] = acceptanceRate
                else
                    if accepted
                        acceptanceRate = ( (1 - (1.0/500)) * acceptanceRate) + (1.0/500)
                    else
                        acceptanceRate = ( (1 - (1.0/500)) * acceptanceRate)
                    end
                    listAcceptanceRate[move] = acceptanceRate
                end
            end
            #
            # end
            #
            
            if move == 40_000
                for w in 1:nWalkers
                    if w_e0[w] + w_dEsum_opt[w] < -5590.0
                        countMiddle += 1
                    end
                end
            end
            
            ####################################################################
            record_T[move] = T
            ####################################################################
            if T <= 1.0 ## stop program immediately if T reaches Tmin=1
                keepLooping = false
                println( "This RURS version stops when temperature reaches 1. It does not wait to complete all moves" )
            end
            #
            #
            if mod( move, tempLength ) == 0
                # before updating with the optimal configs and optimal energies, Elena asked to show standard deviation of
                # the energy at the end of simulation for different runs:
                if nWalkers == 1
                    lastE = w_e0[1] + w_dEsum[1]
                end
                #
                # return to the previous optimal solution after sptesTconstant loop:
                # keepLooking = true
                returnToPreviousOpt!(L, Ne, nWalkers, keepLooping, w_r, w_n, w_r_opt, w_n_opt, isOptimal, w_dEsum_opt, w_dEsum, w_e0, w_T_opt, w_E, move )
            end

            # if mod( move, 500 ) == 0
            #     acceptanceRate = 1 ###@@@@@@@@@@@@@@@@@@@ assumed 1 WALKER!!!! @@@@@@@@
            #     nAccepted = 0  ###@@@@@@@@@@@@@@@@@@@ assumed 1 WALKER!!!! @@@@@@@@
            
            # end

            #
            # # return to the previous optimal solution after sptesTconstant loop:
            # for i in 1:nWalkers
            #     if walkers_r[i] != w_r_opt[i]
            #         # println("oooooooo")
            #         # walkers_r[i] = copy.deepcopy(w_r_opt[i])
            #         walkers_r[i] = copy(w_r_opt[i])
            #         walkers_n[i] = copy(w_n_opt[i])
            #         walker_dEsum[i] = w_dEsum_opt[i]
            #     end
            # end

            # # # remove the energetic walker, and clone randomly one of the others:
            # if nWalkers > 1
            #     # iMax  = np.argmax(walker_dEsum)
            #     _, iMax = findmax(walker_dEsum)
            #     iRand = rand( [i for i in 1:nWalkers if i != iMax] )
            #     #
            #     walkers_r[iMax] = deepcopy(walkers_r[iRand])
            #     walkers_n[iMax] = copy(walkers_n[iRand])
            #     walker_dEsum[iMax] = walker_dEsum[iRand]
            #     w_e0[iMax] = w_e0[iRand]
            # end
            ####################################################################
            ##### not needed, because temperatures are already calculated. To *= alpha # geometric cooling scheme
        end
        #
    end
    #
    #
    # println("RURS finished.")
    return w_r, w_n, w_E, w_acc, record_T, countMiddle, lastE, listAcceptanceRate
end




function RURS_3_scheme1(
    refill::Array{Int64,2},
    notRefill::Array{Int64,1},
    L::Int64,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    Tmin::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    r::Array{Int64,2},
    n::Array{Int64,1},
    displayMessages::Bool
    # nCfgNeighbors::Int
    )
    #
    if displayMessages
        println("****** SIMULATION STAGE ******")
        println("moves to perform: ", steps)
        println("Tmax:             ", Tmax)
        println("alpha:            ", alpha)
        println("walkersBornEqual: ", walkersBornEqual)
        println("tempLength:       ", tempLength)
        println("scheme:           ", scheme)
        println("nWalkers:         ", nWalkers)
    end
    #
    countMiddle = 0
    maximumMoves = steps * tempLength
    #
    w_r, w_n, w_r_opt, w_n_opt,
        w_e0, w_dEsum, w_dEsum_opt, w_E, w_acc, record_T, listT, w_T_opt, isOptimal =
        getInitWalkers2( tempLength, maximumMoves, refill, notRefill, L, UionAionBlist, nWalkers, energyBase )
    #

    To = Tmax
    # alpha = 0.9
    #
    move = 0

    lastAccepted = 0 # assuming just 1 walker @@@@@@@@######@@@@@@@@@
    

    # arrays containig just one element, to avoid memory usage
    p_accepted = zeros(Bool,1) # will contain `accepted`
    p_dEsum = zeros(1) # will contain `dEsum`
    p_dE = zeros(1) # will contain `dE`

    lastE = 0
    #
    
    totalMoves = steps * tempLength
    temperatureList = zeros(totalMoves)
    getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )

    keepLooping = true
    move = 0

    acceptanceRate = 1 ###@@@@@@@@@@@@@@@@@@@ assumed 1 WALKER!!!! @@@@@@@@
    nAccepted = 0  ###@@@@@@@@@@@@@@@@@@@ assumed 1 WALKER!!!! @@@@@@@@
    listAcceptanceRate = zeros(totalMoves)

    # alpha = (Tmin / Tmax) ^ (1 / totalMoves)
    # alpha = (1.0 /800.0 ) ^ (1 / (totalMoves / 4)) #<<<<
    # alpha = (1.0 /800.0 ) ^ (1 / (10_000 / 4))
    # alpha = (1.0 /800.0 ) ^ (1 / (tempLength / 4))

    
    alpha = (Tmin / Tmax) ^ (1 / (4*steps) )

    T = Tmax
    upToMoves = 0
    
    # k = 0.995
    k = 0.9
    # epsilon = 0.01

    # for T in temperatureList
    for _ in 1:totalMoves
        move += 1
        #
        if keepLooping
            # getListT!( To, tempLength, scheme, listT)

            # @timeit to "n1b" for j in 1:tempLength
            # @timeit to "n1c" T = listT[j]

            # if move <= 40_000
            #
            for w in 1:nWalkers
                #
                matrixToSlice!(w, L, Ne, w_r, w_n, r, n)
                #
                p_accepted[1] = false
                p_dEsum[1]    = w_dEsum[w]
                p_dE[1]       = 0.0

                if move > 1
                    Eold = w_E[move - 1, w]
                else
                    Eold = w_e0[w]
                end
                #

                
                nCfgNeighbors = getNeighborsToScan(T)
                # nCfgNeighbors = 1
                # nCfgNeighbors = 2
                # nCfgNeighbors = 10
                # nCfgNeighbors = 100
                # nCfgNeighbors = 40
                # nCfgNeighbors = 100

                # if move > 500
                #     if mod( move, 100 ) < 20
                #         nCfgNeighbors = 100
                #     end
                # end

                # if move > 500
                #     if move < 5000
                #         if rand() < 0.2 # probability of 20%
                #             nCfgNeighbors = 100
                #         end
                #     else
                #         if rand() < 0.4 # probability of 50%
                #             nCfgNeighbors = 100
                #         end
                #     end                        
                # end

                # It will mutate p_accepted, p_dEsum, p_dE:
                move_simulation!( L, Ne, r, n, p_accepted, p_dEsum, p_dE, removedSites, U, UionAionBlist, T, nCfgNeighbors, Eold)
                #
                accepted = p_accepted[1]
                w_dEsum[w] = p_dEsum[1]

                if accepted
                    sliceToMatrix!(w, L, Ne, r, n, w_r, w_n)
                    lastAccepted = move
                    
                    nAccepted += 1
                    #
                    if w_dEsum[w] < w_dEsum_opt[w]
                        w_dEsum_opt[w] = w_dEsum[w]
                        w_T_opt[w] = T # save the temperature in which `opt` was found
                        sliceToMatrix!(w, L, Ne, r, n, w_r_opt, w_n_opt)
                        isOptimal[w] = true


                        optE = w_dEsum_opt[1] ############# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ assuming nWalkers==1 !!!!

                    else
                        isOptimal[w] = false
                    end
                else
                    isOptimal[w] = false
                end
                #
                w_acc[move, w] = accepted
                w_E[move, w] = w_e0[w] + w_dEsum[w]

                
                if move < 500
                    listAcceptanceRate[move] = nAccepted / move
                elseif move == 500
                    acceptanceRate = nAccepted / move
                    listAcceptanceRate[move] = acceptanceRate
                else
                    if accepted
                        acceptanceRate = ( (1 - (1.0/500)) * acceptanceRate) + (1.0/500)
                    else
                        acceptanceRate = ( (1 - (1.0/500)) * acceptanceRate)
                    end
                    listAcceptanceRate[move] = acceptanceRate
                end
            end
            #
            # end
            #
            
            if move == 40_000
                for w in 1:nWalkers
                    if w_e0[w] + w_dEsum_opt[w] < -5590.0
                        countMiddle += 1
                    end
                end
            end
            
            ####################################################################
            record_T[move] = T
            ####################################################################
            if T <= 1.0 ## stop program immediately if T reaches Tmin=1
                keepLooping = false
                println( "This RURS version stops when temperature reaches 1. It does not wait to complete all moves" )
                upToMoves = move
            end
            #
            #
            #
            #
            # mutate tempLength and put numberOfClylesWithoutImprove to zero if numberOfClylesWithoutImprove > 3

            if mod( move, tempLength ) == 0
                # before updating with the optimal configs and optimal energies, Elena asked to show standard deviation of
                # the energy at the end of simulation for different runs:
                if nWalkers == 1
                    lastE = w_e0[1] + w_dEsum[1]
                end

                temp0 = w_dEsum_opt[1]

                #
                # return to the previous optimal solution after sptesTconstant loop:
                # keepLooking = true
                returnToPreviousOpt!(L, Ne, nWalkers, keepLooping, w_r, w_n, w_r_opt, w_n_opt, isOptimal, w_dEsum_opt, w_dEsum, w_e0, w_T_opt, w_E, move )

            end
            
            # if mod( move, 500 ) == 0
            #     acceptanceRate = 1 ###@@@@@@@@@@@@@@@@@@@ assumed 1 WALKER!!!! @@@@@@@@
            #     nAccepted = 0  ###@@@@@@@@@@@@@@@@@@@ assumed 1 WALKER!!!! @@@@@@@@
            
            # end

            #
            # # return to the previous optimal solution after sptesTconstant loop:
            # for i in 1:nWalkers
            #     if walkers_r[i] != w_r_opt[i]
            #         # println("oooooooo")
            #         # walkers_r[i] = copy.deepcopy(w_r_opt[i])
            #         walkers_r[i] = copy(w_r_opt[i])
            #         walkers_n[i] = copy(w_n_opt[i])
            #         walker_dEsum[i] = w_dEsum_opt[i]
            #     end
            # end

            # # # remove the energetic walker, and clone randomly one of the others:
            # if nWalkers > 1
            #     # iMax  = np.argmax(walker_dEsum)
            #     _, iMax = findmax(walker_dEsum)
            #     iRand = rand( [i for i in 1:nWalkers if i != iMax] )
            #     #
            #     walkers_r[iMax] = deepcopy(walkers_r[iRand])
            #     walkers_n[iMax] = copy(walkers_n[iRand])
            #     walker_dEsum[iMax] = walker_dEsum[iRand]
            #     w_e0[iMax] = w_e0[iRand]
            # end
            ####################################################################
            ##### not needed, because temperatures are already calculated. To *= alpha # geometric cooling scheme
        end
        #
        if scheme == "scheme6"
            if mod(move, tempLength) == 0
                T = T / 0.4 #k # reheat
            else
                T = alpha * T
            end

        end


        if move <= 500
            # T = alpha * T
        else

            # if acceptanceRate < 0.4
            #     # k = max(epsilon, k - epsilon)
            #     T = T / k # reheat

            # else
            #     T = alpha * T
            # end

            # if mod( move, tempLength ) == 0
            #     T = Tmax
            # end
            # if move < 50_000
            #     if mod( move, tempLength ) == 0
            #         # T = T / k # reheat
            #         T = T / 0.995
            #         # T = T / 0.8
            #         # T = Tmax / 2
            #         # println("move: ", move)
            #     else
            #         T = alpha * T
            #     end
            # else
            #     # tempLength = 1_000
            #     T = alpha * T


            # end
        end

        
        #
    end
    #
    if keepLooping
        upToMoves = totalMoves
    end
    #
    # println("RURS finished.")
    return w_r, w_n, w_E, w_acc, record_T, countMiddle, lastE, listAcceptanceRate, upToMoves
end


# plotSubplot(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT, move1, move2, y1, y2)
function plotSubplot(
            upToMoves::Int,
            namePlot::String,
            factNumPoints::Int64,
            w_E::Array{Float64,2},
            record_T::Array{Float64,1},
            shouldIplotT::Bool,
            x1::Int,
            x2::Int,
            y1::Float64,
            y2::Float64,
            yTemp1::Float64,
            yTemp2::Float64
        )
    fig, ax1 = PyPlot.subplots()
    myColors = ["k", "r", "g", "m", "y"]
    nWalkers = size(w_E)[2]
    for w in 1:nWalkers
        y = [ w_E[i, w] for i in 1:upToMoves if i % factNumPoints == 0]
        x = 1:length(y)
        # https://stackoverflow.com/questions/55368045/how-to-add-the-markersize-argument-to-a-dataframe-plot
        # instead of `markersize` as in matplotlib.Pyplot, here you have to use `s`:
        if nWalkers <= 5
            ax1.scatter(x, y, s=0.5, color=myColors[w])
        else
            ax1.scatter(x, y, s=0.5)
        end
    end
    ax1.set_ylabel("energy (eV)", color="k")
    ax1.set_ylim([y1, y2]) # zoom # ax1.set_ylim([-5600, -5550]) # zoom
    xlabel = "move"
    ax1.set_xlabel(xlabel)
    # a = x[1]
    # b = x[end]
    # d = (b - a) / 8
    # ax1.set_xlim([ b - d, b ]) # zoom
    ax1.set_xlim([ x1, x2 ]) # zoom

    #
    # if record_T !== nothing
    if shouldIplotT
        ax2 = ax1.twinx()
        # y = [ record_T[i] for i in 1:length(record_T)  ]
        # y = [ record_T[i] for i in 1:upToMoves ]
        y = [ record_T[i] for i in 1:upToMoves if i % factNumPoints == 0 ]
        x = 1:length(y)
        ax2.plot(x, y, "b", linewidth=0.4)
        ax2.set_ylabel("effective temperature (a.u.)", color="b")
        ax2.set_ylim([yTemp1, yTemp2])
        ax2.tick_params(axis="y", colors="blue")
    end
    #fig.savefig("plotWalkers_zoom.png")
    # fig.savefig( string(namePlot, "_zoom") )
    fig.savefig(namePlot)
    PyPlot.close(fig)

end


# plotSubplot(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT, move1, move2, y1, y2)
function plotSubplot(
    upToMoves::Int,
    namePlot::String,
    factNumPoints::Int64,
    w_E::Array{Float64,2},
    record_T::Array{Float64,1},
    shouldIplotT::Bool,
    x1::Int,
    x2::Int,
    y1::Float64,
    y2::Float64,
    yTemp1::Float64,
    yTemp2::Float64
    )
    closeall()

    nWalkers = size(w_E)[2]
    
    # plot(rand(10), seriestype = :scatter, markersize=1.5, markerstrokewidth=0, markerstrokecolor=:blue, markercolor=:blue); v = plot!(twinx(),100rand(10))

    for w in 1:nWalkers
        y = [ w_E[i, w] for i in 1:upToMoves if i % factNumPoints == 0]
        x = 1:length(y)
        # https://stackoverflow.com/questions/55368045/how-to-add-the-markersize-argument-to-a-dataframe-plot
        # instead of `markersize` as in matplotlib.Pyplot, here you have to use `s`:
        if nWalkers <= 5
            plot!( x, y, seriestype = :scatter, markersize=1.5, markerstrokewidth=0, markerstrokecolor=:auto, palette = :Dark2_5, label="")
        else
            plot!( x, y, seriestype = :scatter, markersize=1.5, markerstrokewidth=0, markerstrokecolor=:auto, markercolor=:black, label="")
        end
    end
    plot!(xlabel="move")
    plot!(ylabel="energy (eV)")
    plot!(ylims=(y1, y2)) # zoom # ax1.set_ylim([-5600, -5550]) # zoom
    
    # a = x[1]
    # b = x[end]
    # d = (b - a) / 8
    # ax1.set_xlim([ b - d, b ]) # zoom
    plot!(xlims=(x1, x2)) # zoom

    #
    # if record_T !== nothing
    if shouldIplotT
        # y = [ record_T[i] for i in 1:length(record_T)  ]
        # y = [ record_T[i] for i in 1:upToMoves ]
        y = [ record_T[i] for i in 1:upToMoves if i % factNumPoints == 0 ]
        x = 1:length(y)

        plot!(twinx(), x, y, label="")
        plot!(twinx(), ylabel="effective temperature (a.u.)") 
        f = plot!(twinx(), ylims=(yTemp1, yTemp2))

        #ax2.set_ylabel("effective temperature (a.u.)", color="b")
        # ax2.tick_params(axis="y", colors="blue")
    end
    #fig.savefig("plotWalkers_zoom.png")
    # fig.savefig( string(namePlot, "_zoom") )
    
    savefig(f, namePlot)
    closeall()

end

# plotWalkers2(nWalkers, upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
function plotWalkers2_original(
                # nWalkers::Int,
                upToMoves::Int,
                namePlot::String,
                factNumPoints::Int64,
                w_E::Array{Float64,2},
                record_T::Array{Float64,1},
                shouldIplotT::Bool
                )
    # using PyPlot
    # factNumPoints = 1
    
    PyPlot.close() # it gave me errors saying that other plots were open.

    # l = length(w_E[:, 1])
    # x1     = minimum(w_E[:, 1])
    # x2     = l
    # y1     = -5600
    # y2     = -5150
    # yTemp1 = minimum(record_T)
    # yTemp2 = maximum(record_T)
    # namePlot_ = string(namePlot, "_")
    # shouldIplotT = false
    # plotSubplot(upToMoves, namePlot_, factNumPoints, w_E, record_T, shouldIplotT, x1, x2, y1, y2, yTemp1, yTemp2)


    # a = minimum(w_E[:, 1])
    # b = maximum(w_E[:, 1])
    # d = (b - a) / 8
    # # 
    # x1 = b - d
    # x2 = b
    # y1 = -5600
    # y2 = -5550
    # yTemp1 = 0.0
    # yTemp2 = 2.0
    # namePlot_ = string(namePlot, "_zoom")
    # shouldIplotT = true
    # factNumPoints = 1
    # plotSubplot(upToMoves, namePlot_, factNumPoints, w_E, record_T, shouldIplotT, x1, x2, y1, y2, yTemp1, yTemp2)

    # x1  = 20_000
    # x2  = 60_000
    # y1     = -5600
    # y2     = -5150
    # yTemp1 = minimum(record_T)
    # yTemp2 = maximum(record_T)
    # namePlot_ = string(namePlot, "_zoom2")
    # shouldIplotT = false
    # factNumPoints = 1
    # plotSubplot(upToMoves, namePlot_, factNumPoints, w_E, record_T, shouldIplotT, x1, x2, y1, y2, yTemp1, yTemp2)

    # x1  = 37_500
    # x2  = 38_500
    # y1     = -5600
    # y2     = -5150
    # yTemp1 = minimum(record_T)
    # yTemp2 = maximum(record_T)
    # namePlot_ = string(namePlot, "_zoom3")
    # shouldIplotT = true
    # factNumPoints = 1
    # plotSubplot(upToMoves, namePlot_, factNumPoints, w_E, record_T, shouldIplotT, x1, x2, y1, y2, yTemp1, yTemp2)


    # PyPlot.close(fig)
    fig, ax1 = PyPlot.subplots()
    myColors = ["k", "r", "g", "m", "y"]
    # nWalkers = length(w_E)
    nWalkers = size(w_E)[2]
    y_scope = []
    x_scope = []
    for w in 1:nWalkers
        # y = [ w_E[w][i] for i in 1:length(w_E[w]) if i % factNumPoints == 0 ]
        y = [ w_E[i, w] for i in 1:upToMoves if i % factNumPoints == 0]
        x = 1:length(y)
        y_scope = y[:]
        x_scope = x[:]
        # @test length(y) * nWalkers <= 10000
        # https://stackoverflow.com/questions/55368045/how-to-add-the-markersize-argument-to-a-dataframe-plot
        # instead of `markersize` as in matplotlib.Pyplot, here you have to use `s`:
        if nWalkers <= 5
            ax1.scatter(x, y, s=0.5, color=myColors[w])
        else
            ax1.scatter(x, y, s=0.5)
        end
    end
    ax1.set_ylabel("energy (eV)", color="k")
    ax1.set_ylim([-5600, -5150])
    if factNumPoints == 1
        xlabel = "move"
    else
        xlabel = string("move x ", factNumPoints)
    end
    ax1.set_xlabel(xlabel)
    #
    # if record_T !== nothing

    xT = []
    yT = []
    if length(record_T) >= 1
        yT = [ record_T[i] for i in 1:upToMoves if i % factNumPoints == 0 ]
        xT = 1:length(yT)
    end

    # shouldIplotT = false
    if shouldIplotT
        ax2 = ax1.twinx()
        # y = [ record_T[i] for i in 1:length(record_T) if i % factNumPoints == 0 ]
        # y = [ record_T[i] for i in 1:upToMoves if i % factNumPoints == 0 ]
        # x = 1:length(y)
        # yT_scope = y
        # xT_scope = x
        # @test length(y) * nWalkers <= 10000
        ax2.plot(xT, yT, "b", linewidth=0.4)
        ax2.set_ylabel("effective temperature (a.u.)", color="b")
        ax2.tick_params(axis="y", colors="blue")
    end
    
    #
    # PyPlot.savefig("equilibration.png")
    #fig.savefig("plotWalkers.png")
    fig.savefig(namePlot)
    PyPlot.close(fig)
    
    
    fig, ax1 = PyPlot.subplots()
    myColors = ["k", "r", "g", "m", "y"]
    a = 0
    b = 0
    for w in 1:nWalkers
        # y = [ w_E[w][i] for i in 1:length(w_E[w])  ]
        y = [ w_E[i, w] for i in 1:upToMoves ]
        x = 1:length(y)
        a = x[1]
        b = x[end]    
        # https://stackoverflow.com/questions/55368045/how-to-add-the-markersize-argument-to-a-dataframe-plot
        # instead of `markersize` as in matplotlib.Pyplot, here you have to use `s`:
        if nWalkers <= 5
            ax1.scatter(x, y, s=0.1, color=myColors[w])
        else
            ax1.scatter(x, y, s=0.5)
        end
    end
    ax1.set_ylabel("energy (eV)", color="k")
    ax1.set_ylim([-5600, -5550]) # zoom # ax1.set_ylim([-5600, -5550]) # zoom
    xlabel = "move"
    ax1.set_xlabel(xlabel)
    # a = x[1]
    # b = x[end]
    # d = (b - a) / 8
    d = 5_000 #500
    ax1.set_xlim([ b - d, b ]) # zoom

    #
    # if record_T !== nothing
    if shouldIplotT
        ax2 = ax1.twinx()
        # y = [ record_T[i] for i in 1:length(record_T)  ]
        yT_ = [ record_T[i] for i in 1:upToMoves ]
        xT_ = 1:length(yT_)
        ax2.plot(xT_, yT_, "b", linewidth=0.4)
        ax2.set_ylabel("effective temperature (a.u.)", color="b")
        ax2.set_ylim([0.9, 2.5])
        # ax2.set_ylim([0, 2])
        # ax2.set_ylim([0, 4])
        ax2.tick_params(axis="y", colors="blue")
    end
    #fig.savefig("plotWalkers_zoom.png")
    fig.savefig( string(namePlot, "_zoom") )
    PyPlot.close(fig)

    ###############################################################

    # shouldIplotT = false
    # fig, ax1 = PyPlot.subplots()
    # ax1.scatter(x_scope, y_scope, s=0.5, color="k")
    # ax1.set_xlim([20_000, 60_000]) # zoom # ax1.set_ylim([-5600, -5550]) # zoom
    # ax1.set_ylim([-5600, -5150])
    # ax1.set_ylabel("energy (eV)", color="k")
    # fig.savefig( string(namePlot, "_zoom2") )
    # PyPlot.close(fig)


    fig, ax1 = PyPlot.subplots()
    for w in 1:nWalkers
        # y = [ w_E[w][i] for i in 1:length(w_E[w])  ]
        y = [ w_E[i, w] for i in 1:upToMoves ]
        x = 1:length(y)
        a = x[1]
        b = x[end]    
        # https://stackoverflow.com/questions/55368045/how-to-add-the-markersize-argument-to-a-dataframe-plot
        # instead of `markersize` as in matplotlib.Pyplot, here you have to use `s`:
        if nWalkers <= 5
            ax1.scatter(x, y, s=0.1, color=myColors[w])
        else
            ax1.scatter(x, y, s=0.5)
        end
    end
    ax1.set_xlim([ 30_000, 50_000 ]) # zoom
    ax1.set_ylim([-5600, -5550]) # zoom # ax1.set_ylim([-5600, -5550]) # zoom
    shouldIplotT = true
    if shouldIplotT
        ax2 = ax1.twinx()
        # y = [ record_T[i] for i in 1:length(record_T)  ]
        yT_ = [ record_T[i] for i in 1:upToMoves ]
        xT_ = 1:length(yT_)
        ax2.plot(xT_, yT_, "b", linewidth=0.4)
        ax2.set_ylabel("effective temperature (a.u.)", color="b")
        ax2.set_ylim([0, 110])
        # ax2.set_ylim([0, 2])
        # ax2.set_ylim([0, 4])
        ax2.tick_params(axis="y", colors="blue")
    end
    fig.savefig( string(namePlot, "_zoom3") )
    PyPlot.close(fig)
    



    fig, ax1 = PyPlot.subplots()
    for w in 1:nWalkers
        # y = [ w_E[w][i] for i in 1:length(w_E[w])  ]
        y = [ w_E[i, w] for i in 1:upToMoves ]
        x = 1:length(y)
        a = x[1]
        b = x[end]    
        # https://stackoverflow.com/questions/55368045/how-to-add-the-markersize-argument-to-a-dataframe-plot
        # instead of `markersize` as in matplotlib.Pyplot, here you have to use `s`:
        if nWalkers <= 5
            ax1.scatter(x, y, s=0.1, color=myColors[w])
        else
            ax1.scatter(x, y, s=0.5)
        end
    end
    ax1.set_xlim([ 40_000, 40_400 ]) # zoom
    ax1.set_ylim([-5600, -5550]) # zoom # ax1.set_ylim([-5600, -5550]) # zoom
    shouldIplotT = true
    if shouldIplotT
        ax2 = ax1.twinx()
        # y = [ record_T[i] for i in 1:length(record_T)  ]
        yT_ = [ record_T[i] for i in 1:upToMoves ]
        xT_ = 1:length(yT_)
        ax2.plot(xT_, yT_, "b", linewidth=0.4)
        ax2.set_ylabel("effective temperature (a.u.)", color="b")
        # ax2.set_ylim([54, 55.5])
        ax2.set_ylim([0, 55.5])
        # ax2.set_ylim([0, 2])
        # ax2.set_ylim([0, 4])
        ax2.tick_params(axis="y", colors="blue")
    end
    fig.savefig( string(namePlot, "_zoom4") )
    PyPlot.close(fig)



    PyPlot.close(fig)
    

end

# plotWalkers2(nWalkers, upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
function plotWalkers2(
    # nWalkers::Int,
    upToMoves::Int,
    namePlot::String,
    factNumPoints::Int64,
    w_E::Array{Float64,2},
    record_T::Array{Float64,1},
    shouldIplotT::Bool
    )
    #
    # if nWalkers <= 5
    #     if shouldIplotT
    #         option = "le5_T"
    #     else
    #         option = "le5"
    #     end
    # else
    #     if shouldIplotT
    #         option = "g5_T"
    #     else
    #         option = "g5"
    #     end
    # end
    
    
    
    
    writedlm( "upToMoves.csv",  upToMoves, ',')

    open("namePlot.txt", "w") do io
        write(io, namePlot)
    end
    
    # println(namePlot)
    writedlm( "factNumPoints.csv",  factNumPoints, ',')
    writedlm( "w_E.csv",  w_E, ',')
    writedlm( "record_T.csv",  record_T, ',')
    writedlm( "shouldIplotT.csv",  shouldIplotT, ',')
    println(shouldIplotT)

    run(`python toplot.py`)


end

#  matrixToSlice!(w, L, Ne, w_r, w_n, r, n)
function matrixToSlice!(
            w::Int,
            L::Int,
            Ne::Int,
            w_r::Array{Int64,3},
            w_n::Array{Int64,2},
            r::Array{Int64,2},
            n::Array{Int64,1}
            )
	# matrix to slice:
	for i in 1:L::Int
	    for j in 1:4
	        r[j,i] = w_r[j,i,w]
	    end
	end
	for i in 1:Ne
	    n[i] = w_n[i,w]
	end
end

# sliceToMatrix!(w, L, Ne, r, n, w_r, w_n)
function sliceToMatrix!(
            w::Int,
            L::Int,
            Ne::Int,
            r::Array{Int64,2},
            n::Array{Int64,1},
            w_r::Array{Int64,3},
            w_n::Array{Int64,2}
            )
    #return to matrix:
    for i in 1:L::Int
        for j in 1:4
            w_r[j,i,w] = r[j,i]
        end
    end
    for i in 1:Ne
        w_n[i,w] = n[i]
    end
end

# w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves = equilibration3(maximumMoves, refill, notRefill,...)
function EQ_1(
    refill::Array{Int64,2},
    notRefill::Array{Int64,1},
    L::Int64,
    Ne::Int64,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    energyBase::Float64,
    r::Array{Int64,2},
    n::Array{Int64,1},
    rTemp::Array{Int64,2},
    nTemp::Array{Int64,1},
    rTemp2::Array{Int64,2},
    nTemp2::Array{Int64,1},
    nCfgNeighbors::Int,
    nCheck:: Int,
    displayMessages::Bool
    )
    #
    if displayMessages
        println("****** EQUILIBRATION STAGE ******")
        println("steps to perform: ", steps)
        println("tempLength:       ", tempLength)
        println("moves=steps*tLeng:", steps * tempLength)
        println("Tmax:             ", Tmax)
        println("alpha:            ", alpha)
        println("walkersBornEqual: ", walkersBornEqual)
        println("scheme:           ", scheme)
        println("nWalkers:         ", nWalkers)
    end
    #
    maximumMoves = steps * tempLength
    #
    w_r, w_n, w_r_opt, w_n_opt,
        w_e0, w_dEsum, w_dEsum_opt, w_E, w_acc, record_T, listT, w_T_opt, isOptimal =
        getInitWalkers2( tempLength, maximumMoves, refill, notRefill, L, UionAionBlist, nWalkers, energyBase )
    #
    # w_T     = zeros(nWalkers)
    # @timeit to "a2" w_T_opt = zeros(nWalkers)
    #
    To = Tmax
    # alpha = 0.9
    #
    # @timeit to "a4" record_T = zeros(maximumMoves)
    # nCheck = 100 #50 #100 #500 #1000 #3000
    moves = 0
    keepLooping = true


    # @timeit to "a8" r = zeros(Int, 4, L)
    # @timeit to "a9" n = zeros(Int, Ne)
    # isOptimal = zeros(Bool, nWalkers)
    lastAccepted = 0 # assuming just 1 walker @@@@@@@@######@@@@@@@@@

    # arrays containig just one element, to avoid memory usage
    p_accepted = zeros(Bool,1) # will contain `accepted`
    p_dEsum = zeros(1) # will contain `dEsum`
    p_dE = zeros(1) # will contain `dE`

    for s in 1:steps
        # println("s: ", s, " | ", steps)
        if keepLooping
        # @timeit to "b1" listT = getListT(To, tempLength, scheme)
        getListT!( To, tempLength, scheme, listT)
        for T in listT
            # if T < 1.0
            #     println(".............. ", To, " | ", s, " | ", steps)
            # end

            if keepLooping
                ####################################################################
                # move:
                moves += 1
                for w in 1:nWalkers
                    #
                    # println("j: ", j)
                    # println("moves: ", moves)
                    #
                    matrixToSlice!(w, L, Ne, w_r, w_n, r, n)
                    #
                    p_accepted[1] = false
                    p_dEsum[1]    = w_dEsum[w]
                    p_dE[1]       = 0.0
                    #
                    if moves > 1
                        Eold = w_E[moves - 1, w]
                    else
                        Eold = w_e0[w]
                    end    

                    # nCfgNeighbors = getNeighborsToScan(T)

                    # this will mutate the content of r and n:
                    move_equilibration2!(L, Ne, r, n, p_accepted, p_dEsum, p_dE, removedSites, U, UionAionBlist, L_list, T, rTemp, nTemp, rTemp2, nTemp2, nCfgNeighbors, Eold )
                    #
                    accepted = p_accepted[1]
                    w_dEsum[w] = p_dEsum[1]
                    #
                    sliceToMatrix!(w, L, Ne, r, n, w_r, w_n)
                    #
                    # println("moves: ", moves, " ... ", T, " ... ", accepted)
                    #
                    # println("...", accepted)
                    if accepted::Bool
                    # @timeit to "d1" if p_accepted[1]::Bool
                        lastAccepted = moves
                        if w_dEsum[w] < w_dEsum_opt[w]
                            # ! @@@@@@@@@@@@@@@@@@@@@@@@ 
                            # ! if `refill` is already the optimal one, this conditional will never be evaluated!!! leaving w_T_opt[w]==-1.0
                            # ! @@@@@@@@@@@@@@@@@@@@@@@@
                            w_dEsum_opt[w] = w_dEsum[w]
                            w_T_opt[w] = T # save the temperature in which `opt` was found
                            #
                            sliceToMatrix!(w, L, Ne, r, n, w_r_opt, w_n_opt)
                            isOptimal[w] = true
                            # println("isoptimal ", T)

                        else
                            isOptimal[w] = false
                        end
                    else
                        isOptimal[w] = false
                    end
                    #
                    w_acc[moves, w] = accepted
                    w_E[moves, w] = w_e0[w] + w_dEsum[w]
                end
                ####################################################################
                record_T[moves] = T

                if moves > nCheck
                    if moves - lastAccepted >= nCheck
                        keepLooping = false
                        # println("keepLooping=false ", moves, " ... ", lastAccepted, " ... ", nCheck)
                    end

                    # if sum( w_acc[1][end-nCheck:end] ) == 0
                    #     keepLooping = false
                    #     # println("keepLooping = false")
                    # end
                end
            end
        end
        #
        # return to the previous optimal solution after sptesTconstant loop:
        returnToPreviousOpt!(L, Ne, nWalkers, keepLooping, w_r, w_n, w_r_opt, w_n_opt, isOptimal, w_dEsum_opt, w_dEsum, w_e0, w_T_opt, w_E, moves )

        #
        # # # remove the energetic walker, and clone randomly one of the others:
        # if nWalkers > 1
        #     # iMax  = np.argmax(walker_dEsum)
        #     _, iMax = findmax(walker_dEsum)
        #     iRand = rand( [i for i in 1:nWalkers if i != iMax] )
        #     #
        #     walkers_r[iMax] = deepcopy(walkers_r[iRand])
        #     walkers_n[iMax] = copy(walkers_n[iRand])
        #     walker_dEsum[iMax] = walker_dEsum[iRand]
        #     w_e0[iMax] = w_e0[iRand]
        # end
        ####################################################################
        To *= alpha # geometric cooling scheme
        end
    end
    # println(".............")
    # println("Equilibration finished.")
    for w in 1:nWalkers
        if w_T_opt[w] == -1.0
            println("w_T_opt[w] = -1.0, so it means the initial input was already an optimal one.")
            println("You should run it for a longer time.")
        end
    end
    #
    return w_r, w_n, w_E, w_acc, record_T, w_T_opt, moves
end

# w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves = equilibration3(maximumMoves, refill, notRefill,...)
function EQ_2(
    refill::Array{Int64,2},
    notRefill::Array{Int64,1},
    L::Int64,
    Ne::Int64,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    Tmin::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    energyBase::Float64,
    r::Array{Int64,2},
    n::Array{Int64,1},
    rTemp::Array{Int64,2},
    nTemp::Array{Int64,1},
    rTemp2::Array{Int64,2},
    nTemp2::Array{Int64,1},
    nCfgNeighbors::Int,
    nCheck:: Int,
    displayMessages::Bool
    )
    #
    if displayMessages
        println("****** EQUILIBRATION STAGE ******")
        println("steps to perform: ", steps)
        println("tempLength:       ", tempLength)
        println("moves=steps*tLeng:", steps * tempLength)
        println("Tmax:             ", Tmax)
        println("alpha:            ", alpha)
        println("walkersBornEqual: ", walkersBornEqual)
        println("scheme:           ", scheme)
        println("nWalkers:         ", nWalkers)
    end
    #
    maximumMoves = steps * tempLength
    #
    w_r, w_n, w_r_opt, w_n_opt,
        w_e0, w_dEsum, w_dEsum_opt, w_E, w_acc, record_T, listT, w_T_opt, isOptimal =
        getInitWalkers2( tempLength, maximumMoves, refill, notRefill, L, UionAionBlist, nWalkers, energyBase )
    #
    # w_T     = zeros(nWalkers)
    # @timeit to "a2" w_T_opt = zeros(nWalkers)
    #
    To = Tmax
    # alpha = 0.9
    #
    # @timeit to "a4" record_T = zeros(maximumMoves)
    # nCheck = 100 #50 #100 #500 #1000 #3000
    move = 0
    
    # @timeit to "a8" r = zeros(Int, 4, L)
    # @timeit to "a9" n = zeros(Int, Ne)
    # isOptimal = zeros(Bool, nWalkers)
    lastAccepted = 0 # assuming just 1 walker @@@@@@@@######@@@@@@@@@

    # arrays containig just one element, to avoid memory usage
    p_accepted = zeros(Bool,1) # will contain `accepted`
    p_dEsum = zeros(1) # will contain `dEsum`
    p_dE = zeros(1) # will contain `dE`

    
    totalMoves = steps * tempLength
    temperatureList = zeros(totalMoves)
    getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )

    keepLooping = true
    move = 0

    for T in temperatureList
        move += 1
        #
        if keepLooping
            for w in 1:nWalkers
                #
                # println("j: ", j)
                # println("moves: ", moves)
                #
                matrixToSlice!(w, L, Ne, w_r, w_n, r, n)
                #
                p_accepted[1] = false
                p_dEsum[1]    = w_dEsum[w]
                p_dE[1]       = 0.0
                #
                if move > 1
                    Eold = w_E[move - 1, w]
                else
                    Eold = w_e0[w]
                end    

                # nCfgNeighbors = getNeighborsToScan(T)

                # this will mutate the content of r and n:
                move_equilibration2!(L, Ne, r, n, p_accepted, p_dEsum, p_dE, removedSites, U, UionAionBlist, L_list, T, rTemp, nTemp, rTemp2, nTemp2, nCfgNeighbors, Eold )
                #
                accepted = p_accepted[1]
                w_dEsum[w] = p_dEsum[1]
                #
                sliceToMatrix!(w, L, Ne, r, n, w_r, w_n)
                #
                # println("moves: ", moves, " ... ", T, " ... ", accepted)
                #
                # println("...", accepted)
                if accepted::Bool
                # @timeit to "d1" if p_accepted[1]::Bool
                    lastAccepted = move
                    if w_dEsum[w] < w_dEsum_opt[w]
                        # ! @@@@@@@@@@@@@@@@@@@@@@@@ 
                        # ! if `refill` is already the optimal one, this conditional will never be evaluated!!! leaving w_T_opt[w]==-1.0
                        # ! @@@@@@@@@@@@@@@@@@@@@@@@
                        w_dEsum_opt[w] = w_dEsum[w]
                        w_T_opt[w] = T # save the temperature in which `opt` was found
                        #
                        sliceToMatrix!(w, L, Ne, r, n, w_r_opt, w_n_opt)
                        isOptimal[w] = true
                        # println("isoptimal ", T)

                    else
                        isOptimal[w] = false
                    end
                else
                    isOptimal[w] = false
                end
                #
                w_acc[move, w] = accepted
                w_E[move, w] = w_e0[w] + w_dEsum[w]
            end
            ####################################################################
            record_T[move] = T
            ####################################################################
            if move > nCheck
                if move - lastAccepted >= nCheck
                    keepLooping = false
                    # println("keepLooping=false ", moves, " ... ", lastAccepted, " ... ", nCheck)
                end

                # if sum( w_acc[1][end-nCheck:end] ) == 0
                #     keepLooping = false
                #     # println("keepLooping = false")
                # end
            end
            
            if mod( move, tempLength ) == 0
                # return to the previous optimal solution after sptesTconstant loop:
                returnToPreviousOpt!(L, Ne, nWalkers, keepLooping, w_r, w_n, w_r_opt, w_n_opt, isOptimal, w_dEsum_opt, w_dEsum, w_e0, w_T_opt, w_E, move )
            end
        end
    end

    # println(".............")
    # println("Equilibration finished.")
    for w in 1:nWalkers
        if w_T_opt[w] == -1.0
            println("w_T_opt[w] = -1.0, so it means the initial input was already an optimal one.")
            println("You should run it for a longer time.")
        end
    end
    #
    return w_r, w_n, w_E, w_acc, record_T, w_T_opt, move
end


function experiment11(
    nRepeats::Int,
    L1::Int64,
    L::Int64,
    Ne::Int,
    Nv::Int64,
    ion1::Int,
    ion2::Int,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    walkersBornEqual::Bool,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    nCfgNeighbors::Int
    )
    #
    refill0 = get_refill(L1, L, Nv, ion1, ion2, removedSites)
    refill  = zeros(Int, 4, L)
    myCopyRefill!(L, refill0, refill)
    #
    notRefill0 = get_notRefill(refill, L, Ne, Nv)
    notRefill  = zeros(Int, Ne)
    myCopyNotRefill!(Ne, notRefill0, notRefill)

    # lE = [zeros(0) for w in 1:nRepeats]
    lE = zeros( steps * tempLength, nRepeats )
    record_T_scope = zeros(0) #[]
    #
    r = zeros(Int64, 4, L)
    n = zeros(Int64, Ne)
    rTemp = zeros(Int64, 4, L)
    nTemp = zeros(Int64, Ne)
    rTemp2 = zeros(Int64, 4, L)
    nTemp2 = zeros(Int64, Ne)

    nCheck = 100 #100 #50 #100 #500 #1000 #3000 to be used in `equilibration3()`
    #
    # println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
    for ii in 1:nRepeats
        println("ii: ", ii)
        
        # this will make all `nRepeats` to begin with the same `refill0` configuration
        myCopyRefill!(L, refill0, refill)
        myCopyNotRefill!(Ne, notRefill0, notRefill)

        #
        @time begin
            # alpha2 = (1.0 / Tmax) ^ (1 / steps)
            #
            w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves =
                equilibration3(refill, notRefill, L, Ne, removedSites,
                    UionAionBlist, nWalkers, Tmax, tempLength, scheme,
                    steps, alpha, energyBase, r, n, rTemp, nTemp, rTemp2, nTemp2, nCfgNeighbors, nCheck  )
        end
        # println("Checking energies:")
        for w in 1:nWalkers
            energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,w], U, UionAionBlist )
            println( "energy of opt: ", energy )
        end
        #
        factNumPoints = 10
        namePlot = string("EQ_", string(ii))
        shouldIplotT = true #false
        plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        println("upToMoves: ", upToMoves)
        #
        @time begin
            #
            TmaxEqFound = w_T_opt[1] # <<< you see we are assuming nWalkers=1
            println("TmaxEqFound:: ", TmaxEqFound)
            #
            # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
            w_firstWalker = 1
            for i in 1:L::Int
                for j in 1:4
                    refill[j,i] = w_r[j, i, w_firstWalker]
                end
            end
            for i in 1:Ne
                notRefill[i] = w_n[i, w_firstWalker]
            end
            # now you'll use the new refill and notRefill for the next step


            # alpha2 = (0.1 / TmaxEqFound) ^ (1 / steps)
            #
            w_r, w_n, w_E, w_acc, record_T = simulation( refill, notRefill,
                                    L, removedSites,
                                    UionAionBlist, nWalkers,
                                    TmaxEqFound,
                                    tempLength, scheme, steps, alpha,
                                    r, n,
                                    nCfgNeighbors)
            #
            energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,1], U, UionAionBlist )
            # println(w_r[:,:,1])
            println("energy_: ", energy)

        end
        println("=========")
        println("energy_w_E[end,1]: ", w_E[end, 1])
        #
        factNumPoints= 10
        namePlot = string("RURS_", string(ii))
        shouldIplotT = true #false
        upToMoves = steps * tempLength
        plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        #
        for j in 1:upToMoves
            lE[j, ii] = w_E[j, 1] # `i` index of nRepeats # <<--@@@@@@@@@ assuming just nWalkers=1 !!!!!
        end
        #
    end
    factNumPoints= 50
    namePlot = "RURS_afterEQ"
    shouldIplotT = false
    upToMoves = steps * tempLength
    plotWalkers2(upToMoves, namePlot, factNumPoints, lE, record_T_scope, shouldIplotT)
    # return lE, record_T_scope
end


function experiment12(
    nRepeats::Int,
    L1::Int64,
    L::Int64,
    Ne::Int,
    Nv::Int64,
    ion1::Int,
    ion2::Int,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    walkersBornEqual::Bool,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    nCfgNeighbors::Int
    )
    #
    refill0 = get_refill(L1, L, Nv, ion1, ion2, removedSites)
    refill  = zeros(Int, 4, L)
    myCopyRefill!(L, refill0, refill)
    #
    notRefill0 = get_notRefill(refill, L, Ne, Nv)
    notRefill  = zeros(Int, Ne)
    myCopyNotRefill!(Ne, notRefill0, notRefill)

    # lE = [zeros(0) for w in 1:nRepeats]
    lE = zeros( steps * tempLength, nRepeats )
    record_T_scope = zeros(0) #[]
    #
    r = zeros(Int64, 4, L)
    n = zeros(Int64, Ne)
    rTemp = zeros(Int64, 4, L)
    nTemp = zeros(Int64, Ne)
    rTemp2 = zeros(Int64, 4, L)
    nTemp2 = zeros(Int64, Ne)
    
    # nCheck = 100 #50 #100 #500 #1000 #3000 to be used in `equilibration3()`
    #
    # println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
    for ii in 1:nRepeats
        println("ii: ", ii)
        
        # this will make all `nRepeats` to begin with the same `refill0` configuration
        myCopyRefill!(L, refill0, refill)
        myCopyNotRefill!(Ne, notRefill0, notRefill)

        # #
        # @time begin
        #     # alpha2 = (1.0 / Tmax) ^ (1 / steps)
        #     #
        #     w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves =
        #         equilibration3(refill, notRefill, L, Ne, removedSites,
        #             UionAionBlist, nWalkers, Tmax, tempLength, scheme,
        #             steps, alpha, energyBase, r, n, rTemp, nTemp, rTemp2, nTemp2, nCfgNeighbors, nChecks  )
        # end
        # # println("Checking energies:")
        # for w in 1:nWalkers
        #     energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,w], U, UionAionBlist )
        #     println( "energy of opt: ", energy )
        # end
        # #
        # factNumPoints = 10
        # namePlot = string("EQ_", string(ii))
        # shouldIplotT = true #false
        # plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        # println("upToMoves: ", upToMoves)
        # #
        @time begin
            #
            # TmaxEqFound = w_T_opt[1] # <<< you see we are assuming nWalkers=1
            TmaxEqFound = Tmax
            # println("TmaxEqFound:: ", TmaxEqFound)
            #
            # # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
            # w_firstWalker = 1
            # for i in 1:L::Int
            #     for j in 1:4
            #         refill[j,i] = w_r[j, i, w_firstWalker]
            #     end
            # end
            # for i in 1:Ne
            #     notRefill[i] = w_n[i, w_firstWalker]
            # end
            # # now you'll use the new refill and notRefill for the next step


            # alpha2 = (0.1 / TmaxEqFound) ^ (1 / steps)
            #
            w_r, w_n, w_E, w_acc, record_T = simulation( refill, notRefill,
                                    L, removedSites,
                                    UionAionBlist, nWalkers,
                                    TmaxEqFound,
                                    tempLength, scheme, steps, alpha,
                                    r, n,
                                    nCfgNeighbors)
            #
            energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,1], U, UionAionBlist )
            # println(w_r[:,:,1])
            println("energy_: ", energy)

        end
        println("=========")
        println("energy_w_E[end,1]: ", w_E[end, 1])
        #
        factNumPoints= 10
        namePlot = string("RURS_", string(ii))
        shouldIplotT = true #false
        upToMoves = steps * tempLength
        plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        #
        for j in 1:upToMoves
            lE[j, ii] = w_E[j, 1] # `i` index of nRepeats # <<--@@@@@@@@@ assuming just nWalkers=1 !!!!!
        end
        #
    end
    factNumPoints= 50
    namePlot = "RURS_afterEQ"
    shouldIplotT = false
    upToMoves = steps * tempLength
    plotWalkers2(upToMoves, namePlot, factNumPoints, lE, record_T_scope, shouldIplotT)
    # return lE, record_T_scope
end


function experiment13(
    nRepeats::Int,
    L1::Int64,
    L::Int64,
    Ne::Int,
    Nv::Int64,
    ion1::Int,
    ion2::Int,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    neighbors_EQ::Int,
    nCheck_EQ::Int
    )
    #
    refill0 = get_refill(L1, L, Nv, ion1, ion2, removedSites)
    refill  = zeros(Int, 4, L)
    myCopyRefill!(L, refill0, refill)
    #
    notRefill0 = get_notRefill(refill, L, Ne, Nv)
    notRefill  = zeros(Int, Ne)
    myCopyNotRefill!(Ne, notRefill0, notRefill)

    # lE = [zeros(0) for w in 1:nRepeats]
    lE = zeros( steps * tempLength, nRepeats )
    record_T_scope = zeros(0) #[]
    #
    r = zeros(Int64, 4, L)
    n = zeros(Int64, Ne)
    rTemp = zeros(Int64, 4, L)
    nTemp = zeros(Int64, Ne)
    rTemp2 = zeros(Int64, 4, L)
    nTemp2 = zeros(Int64, Ne)

    # nCheck_EQ = 100 #50 #100 #500 #1000 #3000 to be used in `equilibration3()`

    #
    # println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
    for ii in 1:nRepeats
        println("ii: ", ii)
        
        # this will make all `nRepeats` to begin with the same `refill0` configuration
        myCopyRefill!(L, refill0, refill)
        myCopyNotRefill!(Ne, notRefill0, notRefill)

        #
        @time begin
            # alpha2 = (1.0 / Tmax) ^ (1 / steps)
            #
            println("neighbors_EQ: ", neighbors_EQ)
            println("nCheck_EQ:    ", nCheck_EQ)
            #
            w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves =
                equilibration3(refill, notRefill, L, Ne, removedSites,
                    UionAionBlist, nWalkers, Tmax, tempLength, scheme,
                    steps, alpha, energyBase, r, n, rTemp, nTemp, rTemp2, nTemp2, neighbors_EQ, nCheck_EQ  )
            #
        end
        # println("Checking energies:")
        for w in 1:nWalkers
            energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,w], U, UionAionBlist )
            println( "energy of opt: ", energy )
        end
        #
        factNumPoints = 10
        namePlot = string("EQ_", string(ii))
        shouldIplotT = true #false
        plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        println("upToMoves: ", upToMoves)
        #
        @time begin
            #
            TmaxEqFound = w_T_opt[1] # <<< you see we are assuming nWalkers=1
            println("TmaxEqFound:: ", TmaxEqFound)
            #
            # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
            w_firstWalker = 1
            for i in 1:L::Int
                for j in 1:4
                    refill[j,i] = w_r[j, i, w_firstWalker]
                end
            end
            for i in 1:Ne
                notRefill[i] = w_n[i, w_firstWalker]
            end
            # now you'll use the new refill and notRefill for the next step


            # alpha2 = (0.1 / TmaxEqFound) ^ (1 / steps)
            #
            # w_r, w_n, w_E, w_acc, record_T = simulation( refill, notRefill,
            #                         L, removedSites,
            #                         UionAionBlist, nWalkers,
            #                         TmaxEqFound,
            #                         tempLength, scheme, steps, alpha,
            #                         r, n,
            #                         nCfgNeighbors)

            w_r, w_n, w_E, w_acc, record_T = RURS_1( refill, notRefill,
                                    L, removedSites,
                                    UionAionBlist, nWalkers,
                                    TmaxEqFound,
                                    tempLength, scheme, steps, alpha,
                                    r, n)
            #
            energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,1], U, UionAionBlist )
            # println(w_r[:,:,1])
            println("energy_: ", energy)

        end
        println("=========")
        println("energy_w_E[end,1]: ", w_E[end, 1])
        #
        factNumPoints= 10
        namePlot = string("RURS_", string(ii))
        shouldIplotT = true #false
        upToMoves = steps * tempLength
        plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        #
        for j in 1:upToMoves
            lE[j, ii] = w_E[j, 1] # `i` index of nRepeats # <<--@@@@@@@@@ assuming just nWalkers=1 !!!!!
        end
        #
    end
    factNumPoints= 50
    namePlot = "RURS_afterEQ"
    shouldIplotT = false
    upToMoves = steps * tempLength
    plotWalkers2(upToMoves, namePlot, factNumPoints, lE, record_T_scope, shouldIplotT)
    # return lE, record_T_scope
end

# this saves w_r[1] into refill
# w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)
function w_r_to_refill!(
        L::Int,
        Ne::Int,
        w_r::Array{Int64,3},
        w_n::Array{Int64,2},
        refill::Array{Int64,2},
        notRefill::Array{Int64,1}
        )
    #
    # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
    w_firstWalker = 1
    for i in 1:L
        for j in 1:4
            refill[j,i] = w_r[j, i, w_firstWalker]
        end
    end
    for i in 1:Ne
        notRefill[i] = w_n[i, w_firstWalker]
    end
end


function experiment14(
    nRepeats::Int,
    L1::Int64,
    L::Int64,
    Ne::Int,
    Nv::Int64,
    ion1::Int,
    ion2::Int,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    neighbors_EQ::Int,
    nCheck_EQ::Int,
    displayMessages::Bool
    )
    #
    refill0 = get_refill(L1, L, Nv, ion1, ion2, removedSites)
    refill  = zeros(Int, 4, L)
    myCopyRefill!(L, refill0, refill)
    #
    notRefill0 = get_notRefill(refill, L, Ne, Nv)
    notRefill  = zeros(Int, Ne)
    myCopyNotRefill!(Ne, notRefill0, notRefill)

    # lE = [zeros(0) for w in 1:nRepeats]
    lE = zeros( steps * tempLength, nRepeats )
    record_T_scope = zeros(0) #[]
    #
    r = zeros(Int64, 4, L)
    n = zeros(Int64, Ne)
    rTemp = zeros(Int64, 4, L)
    nTemp = zeros(Int64, Ne)
    rTemp2 = zeros(Int64, 4, L)
    nTemp2 = zeros(Int64, Ne)

    contador = 0

    # nCheck_EQ = 100 #50 #100 #500 #1000 #3000 to be used in `equilibration3()`

    #
    # println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
    for ii in 1:nRepeats
        if displayMessages
            println("ii: ", ii)
        end
        
        # # this will make all `nRepeats` to begin with the same `refill0` configuration
        # myCopyRefill!(L, refill0, refill)
        # myCopyNotRefill!(Ne, notRefill0, notRefill)

        # println("||||||||||||||")
        # println(notRefill)
        # println("||||||||||||||")

        refill    = get_refill(L1, L, Nv, ion1, ion2, removedSites)
        notRefill = get_notRefill(refill, L, Ne, Nv)
        #
        @time begin
            # alpha2 = (1.0 / Tmax) ^ (1 / steps)
            #
            if displayMessages
                println("neighbors_EQ: ", neighbors_EQ)
                println("nCheck_EQ:    ", nCheck_EQ)
            end
            #
            w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves =
                    EQ_1( refill, notRefill, L, Ne, removedSites,
                    UionAionBlist, nWalkers, Tmax, tempLength, scheme,
                    steps, alpha, energyBase, r, n, rTemp, nTemp, rTemp2, nTemp2, neighbors_EQ, nCheck_EQ, displayMessages  )
            # 

        end
        # println("Checking energies:")
        for w in 1:nWalkers
            energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,w], U, UionAionBlist )
            if displayMessages
                println( "energy of opt: ", energy )
            end
        end
        #
        factNumPoints = 10
        namePlot = string("EQ_", string(ii))
        shouldIplotT = true #false
        plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        if displayMessages
            println("upToMoves: ", upToMoves)
        end
        #
        @time begin
            #
            TmaxEqFound = w_T_opt[1] # <<< you see we are assuming nWalkers=1
            println("TmaxEqFound:: ", TmaxEqFound)
            if TmaxEqFound < 0.0
                TmaxEqFound = Tmax
                println("TmaxEqFound was found negative, so refill was already an optimal one in EQ, I am changning to TmaxEqFound=", Tmax)
            end
            # TmaxEqFound = 250.0
            # TmaxEqFound = 350.0
            # TmaxEqFound = 500.0
            TmaxEqFound = 800.0
            # TmaxEqFound = 1000.0
            # TmaxEqFound = 2000.0
            ###

            # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
            w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)
            # now you'll use the new refill and notRefill for the next step

            # println("||||||||||||||--------------------")
            # println(notRefill)
            # println("||||||||||||||--------------------")


            # alpha2 = (0.1 / TmaxEqFound) ^ (1 / steps)
            #
            # w_r, w_n, w_E, w_acc, record_T = simulation( refill, notRefill,
            #                         L, removedSites,
            #                         UionAionBlist, nWalkers,
            #                         TmaxEqFound,
            #                         tempLength, scheme, steps, alpha,
            #                         r, n,
            #                         nCfgNeighbors)

            w_r, w_n, w_E, w_acc, record_T = RURS_1( refill, notRefill,
                                    L, removedSites,
                                    UionAionBlist, nWalkers,
                                    TmaxEqFound,
                                    tempLength, scheme, steps, alpha,
                                    r, n, displayMessages)
            #
            energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,1], U, UionAionBlist )
            # println(w_r[:,:,1])
            if displayMessages
                println("energy_: ", energy)
            end

            if energy < -5590.0
                println("The following config has energy less")
                println("energy_: ", energy)
                println(refill)
                println("  ")
                println(sort(  [ refill[3, i] for i in 1:L ] ))
                println(sort(notRefill))
                contador += 1
            end

        end
        println("times config < -5590: ", contador)
        println("percentage: ", contador / nRepeats * 100, "%")

        # # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
        # w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)

        if displayMessages
            println("=========")
            println("energy_w_E[end,1]: ", w_E[end, 1])
        end
        #
        factNumPoints= 10
        namePlot = string("RURS_", string(ii))
        shouldIplotT = true #false
        upToMoves = steps * tempLength
        plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        #
        for j in 1:upToMoves
            lE[j, ii] = w_E[j, 1] # `ii` index of nRepeats # <<--@@@@@@@@@ assuming just nWalkers=1 !!!!!
            w_E[j, 1] = 0.0
        end
        #
    end
    factNumPoints= 50
    namePlot = "RURS_afterEQ"
    shouldIplotT = false
    upToMoves = steps * tempLength
    plotWalkers2(upToMoves, namePlot, factNumPoints, lE, record_T_scope, shouldIplotT)
    # return lE, record_T_scope
end


function experiment15(
    nRepeats::Int,
    L1::Int64,
    L::Int64,
    Ne::Int,
    Nv::Int64,
    ion1::Int,
    ion2::Int,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    neighbors_EQ::Int,
    nCheck_EQ::Int,
    displayMessages::Bool,
    plot::Bool
    )
    #
    refill0 = get_refill(L1, L, Nv, ion1, ion2, removedSites)
    refill  = zeros(Int, 4, L)
    myCopyRefill!(L, refill0, refill)
    #
    notRefill0 = get_notRefill(refill, L, Ne, Nv)
    notRefill  = zeros(Int, Ne)
    myCopyNotRefill!(Ne, notRefill0, notRefill)

    # lE = [zeros(0) for w in 1:nRepeats]
    lE = zeros( steps * tempLength, nRepeats )
    record_T_scope = zeros(0) #[]
    #
    r = zeros(Int64, 4, L)
    n = zeros(Int64, Ne)
    rTemp = zeros(Int64, 4, L)
    nTemp = zeros(Int64, Ne)
    rTemp2 = zeros(Int64, 4, L)
    nTemp2 = zeros(Int64, Ne)

    contador = 0
    lessEnergies = []
    
    # nCheck_EQ = 100 #50 #100 #500 #1000 #3000 to be used in `equilibration3()`

    #
    # println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
    for ii in 1:nRepeats
        if displayMessages
            println("ii: ", ii)
        end
        
        # # this will make all `nRepeats` to begin with the same `refill0` configuration
        # myCopyRefill!(L, refill0, refill)
        # myCopyNotRefill!(Ne, notRefill0, notRefill)

        # println("||||||||||||||")
        # println(notRefill)
        # println("||||||||||||||")

        refill    = get_refill(L1, L, Nv, ion1, ion2, removedSites)
        notRefill = get_notRefill(refill, L, Ne, Nv)
        #
        # @time begin
        #     # alpha2 = (1.0 / Tmax) ^ (1 / steps)
        #     #
        #     if displayMessages
        #         println("neighbors_EQ: ", neighbors_EQ)
        #         println("nCheck_EQ:    ", nCheck_EQ)
        #     end
        #     #
        #     w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves =
        #             EQ_1( refill, notRefill, L, Ne, removedSites,
        #             UionAionBlist, nWalkers, Tmax, tempLength, scheme,
        #             steps, alpha, energyBase, r, n, rTemp, nTemp, rTemp2, nTemp2, neighbors_EQ, nCheck_EQ, displayMessages  )
        #     # 

        # end
        # # println("Checking energies:")
        # for w in 1:nWalkers
        #     energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,w], U, UionAionBlist )
        #     if displayMessages
        #         println( "energy of opt: ", energy )
        #     end
        # end
        # #
        # factNumPoints = 10
        # namePlot = string("EQ_", string(ii))
        # shouldIplotT = true #false
        # plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
        # if displayMessages
        #     println("upToMoves: ", upToMoves)
        # end
        #
        @time begin
            #
            # TmaxEqFound = w_T_opt[1] # <<< you see we are assuming nWalkers=1
            # println("TmaxEqFound:: ", TmaxEqFound)
            # if TmaxEqFound < 0.0
            #     TmaxEqFound = Tmax
            #     println("TmaxEqFound was found negative, so refill was already an optimal one in EQ, I am changning to TmaxEqFound=", Tmax)
            # end
            # TmaxEqFound = 250.0
            # TmaxEqFound = 350.0
            # TmaxEqFound = 500.0
            # TmaxEqFound = 1000.0
            # TmaxEqFound = 2000.0
            TmaxEqFound = 800.0
            ###

            # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
            # w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)
            # now you'll use the new refill and notRefill for the next step

            # println("||||||||||||||--------------------")
            # println(notRefill)
            # println("||||||||||||||--------------------")


            # alpha2 = (0.1 / TmaxEqFound) ^ (1 / steps)
            #
            # w_r, w_n, w_E, w_acc, record_T = simulation( refill, notRefill,
            #                         L, removedSites,
            #                         UionAionBlist, nWalkers,
            #                         TmaxEqFound,
            #                         tempLength, scheme, steps, alpha,
            #                         r, n,
            #                         nCfgNeighbors)

            w_r, w_n, w_E, w_acc, record_T = RURS_1( refill, notRefill,
                                    L, removedSites,
                                    UionAionBlist, nWalkers,
                                    TmaxEqFound,
                                    tempLength, scheme, steps, alpha,
                                    r, n, displayMessages)
            #
            energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,1], U, UionAionBlist )
            # println(w_r[:,:,1])
            if displayMessages
                println("energy_: ", energy)
            end

            if energy < -5590.0
                println("The following config has energy less")
                println("energy_: ", energy)
                append!(lessEnergies, energy)                
                # println(refill)
                # println("  ")
                # println(sort(  [ refill[3, i] for i in 1:L ] ))
                # println(sort(notRefill))
                contador += 1
            end

        end
        println("times config < -5590: ", contador)
        println("percentage: ", contador / nRepeats * 100, "%")

        # # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
        # w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)

        if displayMessages
            println("=========")
            println("energy_w_E[end,1]: ", w_E[end, 1])
        end
        #
        factNumPoints= 10
        namePlot = string("RURS_", string(ii))
        shouldIplotT = true #false
        upToMoves = steps * tempLength
        # upToMoves = 40_000
        if nRepeats <= 5
            if plot
                plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
            end
        end
        #
        for j in 1:upToMoves
            lE[j, ii] = w_E[j, 1] # `ii` index of nRepeats # <<--@@@@@@@@@ assuming just nWalkers=1 !!!!!
            w_E[j, 1] = 0.0
        end
        #
    end
    factNumPoints= 50
    namePlot = "RURS_afterEQ"
    shouldIplotT = false
    upToMoves = steps * tempLength
    if nRepeats <= 5
        if plot
            plotWalkers2(upToMoves, namePlot, factNumPoints, lE, record_T_scope, shouldIplotT)
        end
    end
    # return lE, record_T_scope
    return contador, lessEnergies
end


function experiment16(
    nRepeats::Int,
    L1::Int64,
    L::Int64,
    Ne::Int,
    Nv::Int64,
    ion1::Int,
    ion2::Int,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    neighbors_EQ::Int,
    nCheck_EQ::Int,
    displayMessages::Bool
    )
    #
    refill0 = get_refill(L1, L, Nv, ion1, ion2, removedSites)
    refill  = zeros(Int, 4, L)
    myCopyRefill!(L, refill0, refill)
    #
    notRefill0 = get_notRefill(refill, L, Ne, Nv)
    notRefill  = zeros(Int, Ne)
    myCopyNotRefill!(Ne, notRefill0, notRefill)

    # lE = [zeros(0) for w in 1:nRepeats]
    lE = zeros( steps * tempLength, nRepeats )
    record_T_scope = zeros(0) #[]
    #
    r = zeros(Int64, 4, L)
    n = zeros(Int64, Ne)
    rTemp = zeros(Int64, 4, L)
    nTemp = zeros(Int64, Ne)
    rTemp2 = zeros(Int64, 4, L)
    nTemp2 = zeros(Int64, Ne)

    contador_countMiddle = 0
    contador = 0

    # nCheck_EQ = 100 #50 #100 #500 #1000 #3000 to be used in `equilibration3()`

    #
    # println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
    saveToAvg = zeros(nRepeats)
    #
    # deleting previous content:
    io = open("xtest.csv", "w")
    close(io)
    v = zeros(Real,9)
    for _ in 1:5 #nGames=5
        #
        for ii in 1:nRepeats
            if displayMessages
                println("ii: ", ii)
            end
            
            # # this will make all `nRepeats` to begin with the same `refill0` configuration
            myCopyRefill!(L, refill0, refill)
            myCopyNotRefill!(Ne, notRefill0, notRefill)

            # println("||||||||||||||")
            # println(notRefill)
            # println("||||||||||||||")

            # refill    = get_refill(L1, L, Nv, ion1, ion2, removedSites)
            # notRefill = get_notRefill(refill, L, Ne, Nv)
            #
            @time begin
                # alpha2 = (1.0 / Tmax) ^ (1 / steps)
                #
                if displayMessages
                    println("neighbors_EQ: ", neighbors_EQ)
                    println("nCheck_EQ:    ", nCheck_EQ)
                end
                #
                Tmax = 800.0
                w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves =
                        EQ_1( refill, notRefill, L, Ne, removedSites,
                        UionAionBlist, nWalkers, Tmax, tempLength, scheme,
                        steps, alpha, energyBase, r, n, rTemp, nTemp, rTemp2, nTemp2, neighbors_EQ, nCheck_EQ, displayMessages  )
                # 

            end
            # println("Checking energies:")
            for w in 1:nWalkers
                energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,w], U, UionAionBlist )
                if displayMessages
                    println( "energy of opt: ", energy )
                end
            end
            # #
            factNumPoints = 10
            namePlot = string("EQ_", string(ii))
            shouldIplotT = true #false
            if nRepeats <= 5
                plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
            end
            if displayMessages
                println("upToMoves: ", upToMoves)
            end
            #
            @time begin
                #
                # TmaxEqFound = w_T_opt[1] # <<< you see we are assuming nWalkers=1
                # println("TmaxEqFound:: ", TmaxEqFound)
                # if TmaxEqFound < 0.0
                #     TmaxEqFound = Tmax
                #     println("TmaxEqFound was found negative, so refill was already an optimal one in EQ, I am changning to TmaxEqFound=", Tmax)
                # end
                # TmaxEqFound = 250.0
                # TmaxEqFound = 350.0
                # TmaxEqFound = 500.0
                # TmaxEqFound = 1000.0
                # TmaxEqFound = 2000.0
                TmaxEqFound = 800.0
                ###

                # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
                w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)
                # now you'll use the new refill and notRefill for the next step

                # println("||||||||||||||--------------------")
                # println(notRefill)
                # println("||||||||||||||--------------------")


                # alpha2 = (0.1 / TmaxEqFound) ^ (1 / steps)
                #
                w_r, w_n, w_E, w_acc, record_T, countMiddle, lastE = RURS_1( refill, notRefill,
                                        L, removedSites,
                                        UionAionBlist, nWalkers,
                                        TmaxEqFound,
                                        tempLength, scheme, steps, alpha,
                                        r, n, displayMessages)
                #
                contador_countMiddle += countMiddle
                saveToAvg[ii] = lastE
                #
                energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,1], U, UionAionBlist )
                # println(w_r[:,:,1])
                if displayMessages
                    println("energy_: ", energy)
                end

                if energy < -5590.0
                    println("The following config has energy less")
                    println("energy_: ", energy)
                    # println(refill)
                    # println("  ")
                    # println(sort(  [ refill[3, i] for i in 1:L ] ))
                    # println(sort(notRefill))
                    contador += 1
                end
                

            end
            println("times config < -5590 middle: ", contador_countMiddle)
            println("times config < -5590       : ", contador)
            println("percentage: ", contador / nRepeats * 100, "%")

            # # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
            # w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)

            if displayMessages
                println("=========")
                println("energy_w_E[end,1]: ", w_E[end, 1])
            end
            #
            factNumPoints= 10
            namePlot = string("RURS_", string(ii))
            shouldIplotT = true #false
            upToMoves = steps * tempLength
            # upToMoves = 40_000
            record_T_scope = record_T[:]
            if nRepeats <= 5
                plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
            end
            #
            for j in 1:upToMoves
                lE[j, ii] = w_E[j, 1] # `ii` index of nRepeats # <<--@@@@@@@@@ assuming just nWalkers=1 !!!!!
                w_E[j, 1] = 0.0
            end
            #
        end
        factNumPoints= 10
        namePlot = "RURS_afterEQ"
        # shouldIplotT = false
        shouldIplotT = true
        upToMoves = steps * tempLength
        if nRepeats <= 5
            plotWalkers2(upToMoves, namePlot, factNumPoints, lE, record_T_scope, shouldIplotT)
        end
        # return lE, record_T_scope
        #
        println("---------------------------------------")
        println(saveToAvg)
        # run( `echo "HOLA" >> tempResults.txt` )
        # io = open("tempResults.txt", "a")
        # writedlm(io, saveToAvg)
        # close(io)

        if nRepeats == 5
            v[6] = round( mean(saveToAvg), digits=1 )
            v[7] = round( std(saveToAvg), digits=1 )
            for l in 1:5
                v[l] = round( saveToAvg[l], digits=1 )
            end
            v[8] = contador_countMiddle
            v[9] = contador
            io = open("xtest.csv", "a")
            writedlm(io, transpose(v), ',')
            close(io)
        end
        println("average over runs: ", mean(saveToAvg))
        println("std dev over runs: ", std(saveToAvg) )
        println("---------------------------------------.")
    #
    end
    return contador
end

### new
function experiment17(
    nRepeats::Int,
    L1::Int64,
    L::Int64,
    Ne::Int,
    Nv::Int64,
    ion1::Int,
    ion2::Int,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    neighbors_EQ::Int,
    nCheck_EQ::Int,
    displayMessages::Bool
    )
    #
    # deleting previous content:
    io = open("xtest.csv", "w")
    close(io)
    v = zeros(Real,9)
    
    contador_countMiddle = 0
    contador = 0

    for _ in 1:1 #nGames=5
        refill0 = get_refill(L1, L, Nv, ion1, ion2, removedSites)
        refill  = zeros(Int, 4, L)
        myCopyRefill!(L, refill0, refill)
        #
        notRefill0 = get_notRefill(refill, L, Ne, Nv)
        notRefill  = zeros(Int, Ne)
        myCopyNotRefill!(Ne, notRefill0, notRefill)
    
        # lE = [zeros(0) for w in 1:nRepeats]
        lE = zeros( steps * tempLength, nRepeats )
        record_T_scope = zeros(0) #[]
        #
        r = zeros(Int64, 4, L)
        n = zeros(Int64, Ne)
        rTemp = zeros(Int64, 4, L)
        nTemp = zeros(Int64, Ne)
        rTemp2 = zeros(Int64, 4, L)
        nTemp2 = zeros(Int64, Ne)
        
        # nCheck_EQ = 100 #50 #100 #500 #1000 #3000 to be used in `equilibration3()`
    
        #
        # println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
        saveToAvg = zeros(nRepeats)
        #
    
        #
        for ii in 1:nRepeats
            if displayMessages
                println("ii: ", ii)
            end
            
            # # this will make all `nRepeats` to begin with the same `refill0` configuration
            myCopyRefill!(L, refill0, refill)
            myCopyNotRefill!(Ne, notRefill0, notRefill)

            # println("||||||||||||||")
            # println(notRefill)
            # println("||||||||||||||")

            # refill    = get_refill(L1, L, Nv, ion1, ion2, removedSites)
            # notRefill = get_notRefill(refill, L, Ne, Nv)
            #
            @time begin
                # alpha2 = (1.0 / Tmax) ^ (1 / steps)
                #
                if displayMessages
                    println("neighbors_EQ: ", neighbors_EQ)
                    println("nCheck_EQ:    ", nCheck_EQ)
                end
                #
                Tmax = 800.0
                Tmin = 1.0
                w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves =
                        EQ_2( refill, notRefill, L, Ne, removedSites,
                        UionAionBlist, nWalkers, Tmax, Tmin, tempLength, scheme,
                        steps, alpha, energyBase, r, n, rTemp, nTemp, rTemp2, nTemp2, neighbors_EQ, nCheck_EQ, displayMessages  )
                # 

            end
            # println("Checking energies:")
            for w in 1:nWalkers
                energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,w], U, UionAionBlist )
                if displayMessages
                    println( "energy of opt: ", energy )
                end
            end
            # #
            factNumPoints = 10
            namePlot = string("EQ_", string(ii))
            shouldIplotT = true #false
            if nRepeats <= 5
                plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
            end
            if displayMessages
                println("upToMoves: ", upToMoves)
            end
            #
            @time begin
                #
                # TmaxEqFound = w_T_opt[1] # <<< you see we are assuming nWalkers=1
                # println("TmaxEqFound:: ", TmaxEqFound)
                # if TmaxEqFound < 0.0
                #     TmaxEqFound = Tmax
                #     println("TmaxEqFound was found negative, so refill was already an optimal one in EQ, I am changning to TmaxEqFound=", Tmax)
                # end
                # TmaxEqFound = 250.0
                # TmaxEqFound = 350.0
                # TmaxEqFound = 500.0
                # TmaxEqFound = 1000.0
                # TmaxEqFound = 2000.0
                TmaxEqFound = 800.0
                ###

                # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
                w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)
                # now you'll use the new refill and notRefill for the next step

                # println("||||||||||||||--------------------")
                # println(notRefill)
                # println("||||||||||||||--------------------")


                # alpha2 = (0.1 / TmaxEqFound) ^ (1 / steps)
                #
                Tmin = 1.0
                w_r, w_n, w_E, w_acc, record_T, countMiddle,
                    lastE, listAcceptanceRate, 
                    upToMoves = RURS_3_scheme1( refill, notRefill,
                                    L, removedSites,
                                    UionAionBlist, nWalkers,
                                    TmaxEqFound, Tmin,
                                    tempLength, scheme, steps, alpha,
                                    r, n, displayMessages)

                #
                # lastE, listAcceptanceRate = RURS_2( refill, notRefill,
                #                         L, removedSites,
                #                         UionAionBlist, nWalkers,
                #                         TmaxEqFound, Tmin,
                #                         tempLength, scheme, steps, alpha,
                #                         r, n, displayMessages)
                #            
                #
                println("upTomoves rurs: ", upToMoves)
                contador_countMiddle += countMiddle
                saveToAvg[ii] = lastE
                #
                energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,1], U, UionAionBlist )
                # println(w_r[:,:,1])
                if displayMessages
                    println("energy_: ", energy)
                end

                if energy < -5590.0
                    println("The following config has energy less")
                    println("energy_: ", energy)
                    # println(refill)
                    # println("  ")
                    # println(sort(  [ refill[3, i] for i in 1:L ] ))
                    # println(sort(notRefill))
                    contador += 1
                end
                
                if nRepeats == 1
                    if ( contador_countMiddle > 0 ) || ( contador > 0 )
                        io = open("xlistE.txt", "w")
                        writedlm(io, w_E[:, 1])
                        close(io)
                    end
                end


            end
            println("times config < -5590 middle: ", contador_countMiddle)
            println("times config < -5590       : ", contador)
            println("percentage: ", contador / nRepeats * 100, "%")

            # # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
            # w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)

            if displayMessages
                println("=========")
                println("energy_w_E[end,1]: ", w_E[end, 1])
            end
            #
            factNumPoints= 1
            namePlot = string("RURS_", string(ii))
            shouldIplotT = true #false
            # upToMoves = steps * tempLength
            # upToMoves = 40_000
            record_T_scope = record_T[:]
            if nRepeats <= 5
                plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
            end

            factNumPoints= 1
            namePlot = string("RURS_accRate", string(ii))
            shouldIplotT = true #false
            # upToMoves = steps * tempLength
            # upToMoves = 40_000
            if nRepeats <= 5
                plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, listAcceptanceRate, shouldIplotT)
            end

            


            #
            for j in 1:upToMoves
                lE[j, ii] = w_E[j, 1] # `ii` index of nRepeats # <<--@@@@@@@@@ assuming just nWalkers=1 !!!!!
                w_E[j, 1] = 0.0
            end
            #
        end
        factNumPoints= 10
        namePlot = "RURS_afterEQ"
        # shouldIplotT = false
        shouldIplotT = true
        upToMoves = steps * tempLength
        if nRepeats <= 5
            plotWalkers2(upToMoves, namePlot, factNumPoints, lE, record_T_scope, shouldIplotT)
        end
        # return lE, record_T_scope
        #
        println("---------------------------------------")
        println(saveToAvg)
        # run( `echo "HOLA" >> tempResults.txt` )
        # io = open("tempResults.txt", "a")
        # writedlm(io, saveToAvg)
        # close(io)

        if nRepeats == 5
            v[6] = round( mean(saveToAvg), digits=1 )
            v[7] = round( std(saveToAvg), digits=1 )
            for l in 1:5
                v[l] = round( saveToAvg[l], digits=1 )
            end
            v[8] = contador_countMiddle
            v[9] = contador
            io = open("xtest.csv", "a")
            writedlm(io, transpose(v), ',')
            close(io)
        end
        
        println("average over runs: ", mean(saveToAvg))
        println("std dev over runs: ", std(saveToAvg) )
        println("---------------------------------------.")
    #
    end
    return contador
end


### new
function experiment18(
    nRepeats::Int,
    L1::Int64,
    L::Int64,
    Ne::Int,
    Nv::Int64,
    ion1::Int,
    ion2::Int,
    removedSites::Array{Int,1},
    UionAionBlist::Array{Float64,4},
    nWalkers::Int64,
    Tmax::Float64,
    tempLength::Int64,
    scheme::String,
    steps::Int64,
    alpha::Float64,
    neighbors_EQ::Int,
    nCheck_EQ::Int,
    displayMessages::Bool
    )
    #
    # deleting previous content:
    io = open("xtest.csv", "w")
    close(io)
    v = zeros(Real,9)
    
    contador_countMiddle = 0
    contador = 0

    for _ in 1:1 #nGames=5
        refill0 = get_refill(L1, L, Nv, ion1, ion2, removedSites)
        refill  = zeros(Int, 4, L)
        myCopyRefill!(L, refill0, refill)
        #
        notRefill0 = get_notRefill(refill, L, Ne, Nv)
        notRefill  = zeros(Int, Ne)
        myCopyNotRefill!(Ne, notRefill0, notRefill)
    
        # lE = [zeros(0) for w in 1:nRepeats]
        lE = zeros( steps * tempLength, nRepeats )
        record_T_scope = zeros(0) #[]
        #
        r = zeros(Int64, 4, L)
        n = zeros(Int64, Ne)
        rTemp = zeros(Int64, 4, L)
        nTemp = zeros(Int64, Ne)
        rTemp2 = zeros(Int64, 4, L)
        nTemp2 = zeros(Int64, Ne)
        
        # nCheck_EQ = 100 #50 #100 #500 #1000 #3000 to be used in `equilibration3()`
    
        #
        # println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
        saveToAvg = zeros(nRepeats)
        #
    
        #
        for ii in 1:nRepeats
            if displayMessages
                println("ii: ", ii)
            end
            
            # # this will make all `nRepeats` to begin with the same `refill0` configuration
            myCopyRefill!(L, refill0, refill)
            myCopyNotRefill!(Ne, notRefill0, notRefill)

            # println("||||||||||||||")
            # println(notRefill)
            # println("||||||||||||||")

            # refill    = get_refill(L1, L, Nv, ion1, ion2, removedSites)
            # notRefill = get_notRefill(refill, L, Ne, Nv)
            #
            @time begin
                # alpha2 = (1.0 / Tmax) ^ (1 / steps)
                #
                if displayMessages
                    println("neighbors_EQ: ", neighbors_EQ)
                    println("nCheck_EQ:    ", nCheck_EQ)
                end
                #
                Tmax = 800.0
                Tmin = 1.0
                w_r, w_n, w_E, w_acc, record_T, w_T_opt, upToMoves =
                        EQ_2( refill, notRefill, L, Ne, removedSites,
                        UionAionBlist, nWalkers, Tmax, Tmin, tempLength, scheme,
                        steps, alpha, energyBase, r, n, rTemp, nTemp, rTemp2, nTemp2, neighbors_EQ, nCheck_EQ, displayMessages  )
                # 

            end
            # println("Checking energies:")
            for w in 1:nWalkers
                energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,w], U, UionAionBlist )
                if displayMessages
                    println( "energy of opt: ", energy )
                end
            end
            # #
            factNumPoints = 10
            namePlot = string("EQ_", string(ii))
            shouldIplotT = true #false
            if nRepeats <= 5
                #plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
            end
            if displayMessages
                println("upToMoves: ", upToMoves)
            end
            #
            @time begin
                #
                # TmaxEqFound = w_T_opt[1] # <<< you see we are assuming nWalkers=1
                # println("TmaxEqFound:: ", TmaxEqFound)
                # if TmaxEqFound < 0.0
                #     TmaxEqFound = Tmax
                #     println("TmaxEqFound was found negative, so refill was already an optimal one in EQ, I am changning to TmaxEqFound=", Tmax)
                # end
                # TmaxEqFound = 250.0
                # TmaxEqFound = 350.0
                # TmaxEqFound = 500.0
                # TmaxEqFound = 1000.0
                # TmaxEqFound = 2000.0
                TmaxEqFound = 800.0
                ###

                # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
                w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)
                # now you'll use the new refill and notRefill for the next step

                # println("||||||||||||||--------------------")
                # println(notRefill)
                # println("||||||||||||||--------------------")


                # alpha2 = (0.1 / TmaxEqFound) ^ (1 / steps)
                #
                Tmin = 1.0
                w_r, w_n, w_E, w_acc, record_T, countMiddle,
                    lastE, listAcceptanceRate, 
                    upToMoves = RURS_3_scheme1( refill, notRefill,
                                    L, removedSites,
                                    UionAionBlist, nWalkers,
                                    TmaxEqFound, Tmin,
                                    tempLength, scheme, steps, alpha,
                                    r, n, displayMessages)

                #
                # lastE, listAcceptanceRate = RURS_2( refill, notRefill,
                #                         L, removedSites,
                #                         UionAionBlist, nWalkers,
                #                         TmaxEqFound, Tmin,
                #                         tempLength, scheme, steps, alpha,
                #                         r, n, displayMessages)
                #            
                #
                println("upTomoves rurs: ", upToMoves)
                contador_countMiddle += countMiddle
                saveToAvg[ii] = lastE
                #
                energy = getWalkerTotalEnergy( L, energyBase, w_r[:,:,1], U, UionAionBlist )
                # println(w_r[:,:,1])
                if displayMessages
                    println("energy_: ", energy)
                end

                if energy < -5590.0
                    println("The following config has energy less")
                    println("energy_: ", energy)
                    # println(refill)
                    # println("  ")
                    # println(sort(  [ refill[3, i] for i in 1:L ] ))
                    # println(sort(notRefill))
                    contador += 1
                end
                
                if nRepeats == 1
                    if ( contador_countMiddle > 0 ) || ( contador > 0 )
                        io = open("xlistE.txt", "w")
                        writedlm(io, w_E[:, 1])
                        close(io)
                    end
                end


            end
            println("times config < -5590 middle: ", contador_countMiddle)
            println("times config < -5590       : ", contador)
            println("percentage: ", contador / nRepeats * 100, "%")

            # # copy w_r[1] to refill # <<< you see we are assuming nWalkers=1
            # w_r_to_refill!(L, Ne, w_r, w_n, refill, notRefill)

            if displayMessages
                println("=========")
                println("energy_w_E[end,1]: ", w_E[end, 1])
            end
            #
            factNumPoints= 1
            namePlot = string("RURS_", string(ii))
            shouldIplotT = true #false
            # upToMoves = steps * tempLength
            # upToMoves = 40_000
            record_T_scope = record_T[:]
            if nRepeats <= 5
                #plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, record_T, shouldIplotT)
            end

            factNumPoints= 1
            namePlot = string("RURS_accRate", string(ii))
            shouldIplotT = true #false
            # upToMoves = steps * tempLength
            # upToMoves = 40_000
            if nRepeats <= 5
                # plotWalkers2(upToMoves, namePlot, factNumPoints, w_E, listAcceptanceRate, shouldIplotT)
            end

            


            #
            for j in 1:upToMoves
                lE[j, ii] = w_E[j, 1] # `ii` index of nRepeats # <<--@@@@@@@@@ assuming just nWalkers=1 !!!!!
                w_E[j, 1] = 0.0
            end
            #
        end
        factNumPoints= 10
        namePlot = "RURS_afterEQ"
        # shouldIplotT = false
        shouldIplotT = true
        upToMoves = steps * tempLength
        if nRepeats <= 5
            #plotWalkers2(upToMoves, namePlot, factNumPoints, lE, record_T_scope, shouldIplotT)
        end
        # return lE, record_T_scope
        #
        println("---------------------------------------")
        println(saveToAvg)
        # run( `echo "HOLA" >> tempResults.txt` )
        # io = open("tempResults.txt", "a")
        # writedlm(io, saveToAvg)
        # close(io)

        if nRepeats == 5
            v[6] = round( mean(saveToAvg), digits=1 )
            v[7] = round( std(saveToAvg), digits=1 )
            for l in 1:5
                v[l] = round( saveToAvg[l], digits=1 )
            end
            v[8] = contador_countMiddle
            v[9] = contador
            io = open("xtest.csv", "a")
            writedlm(io, transpose(v), ',')
            close(io)
        end
        
        println("average over runs: ", mean(saveToAvg))
        println("std dev over runs: ", std(saveToAvg) )
        println("---------------------------------------.")
    #
    end
    return contador
end


# getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )
function getTemperatureList!(
    Tmax::Float64,
    Tmin::Float64,
    totalMoves::Int,
    steps::Int,
    tempLength::Int64,
    temperatureList::Array{Float64,1},
    scheme::String
    )
    #    
    if scheme == "scheme1"
        alpha = (Tmin / Tmax) ^ (1 / totalMoves)
        T_ = Tmax
        for i in 1:totalMoves
            temperatureList[i] = T_
            T_ *= alpha
        end
    elseif scheme == "scheme2" # constant
        alpha = (Tmin / Tmax) ^ (1 / steps)
        T__ = Tmax
        for i in 1:steps
            # println("i: ", i)
            for j in 1:tempLength
                k = ( (i - 1) * tempLength ) + j
                # println("j, k: ", j, "  ", k)
                temperatureList[k] = T__
            end
            T__ *= alpha # this will be the esqueleton
        end
    elseif scheme == "scheme3" # I think this is Elena's first idea
        alpha = (Tmin / Tmax) ^ (1 / steps)
        #
        for i in 1:steps
            T_ = Tmax
            #
            newAlpha = ( (alpha) ^ i ) ^ (1 / tempLength) # rate between Tprevious and Tnext is alpha!
            for j in 1:tempLength
                k = ( (i - 1) * tempLength ) + j
                temperatureList[k] = T_
                T_ *= newAlpha
            end
        end


        # alpha = (Tmin / Tmax) ^ (1 / steps)
        # T__ = Tmax
        # for i in 1:steps
        #     #
        #     # reheating:
        #     if (i > 1)
        #         T_ = T__ / 0.8 # k=0.8
        #     else
        #         T_ = T__
        #     end
        #     #
        #     for j in 1:tempLength
        #         k = ( (i - 1) * steps ) + j
        #         temperatureList[k] = T_
        #         T_ = alpha * T_
        #     end
        #     #
        #     T__ = alpha * T__ # this will be the esqueleton
        # end
    elseif scheme == "scheme4" # this is what I understood about Elena's second idea
        alpha = (Tmin / Tmax) ^ (1 / steps)
        T__ = Tmax
        Tsaved = Tmax
        T_ = Tmax
        newAlpha = ( (alpha)^2 ) ^ (1 / (2 * tempLength)) # rate between Tprevious and Tnext is alpha!
        for j in 1: (2 * tempLength)
            k = j
            temperatureList[k] = T_
            T_ *= newAlpha
        end
        #
        for i in 3:steps
            T_ = Tsaved
            #
            # newAlpha = ( alpha ) ^ (1 / (0.5 * tempLength)) # rate between Tprevious and Tnext is alpha!
            newAlpha = ( (alpha)^2 ) ^ (1 / ( tempLength)) # rate between Tprevious and Tnext is alpha!
            for j in 1:tempLength
                k = ( (i - 1) * tempLength ) + j
                temperatureList[k] = T_
                # T_ = alpha * T_
                T_ *= newAlpha
            end
            #
            T__ *= alpha # this will be the esqueleton
            Tsaved = T__
        end
    elseif scheme == "scheme5"
        alpha = (Tmin / Tmax) ^ (1 / steps)
        T__ = Tmax
        for i in 1:steps
            # println("i: ", i)
            To = T__
            Tf = T__ / 10
            dT = (Tf - To) / tempLength
            for j in 1:tempLength
                k = ( (i - 1) * tempLength ) + j
                temperatureList[k] = To + ( (j - 1) * dT)
            end
            T__ *= alpha # this will be the esqueleton
        end
    # elseif scheme == "scheme6"
    #     alpha = (Tmin / Tmax) ^ (1 / steps)
    #     T_ = Tmax
    #     for i in 1:totalMoves
    #         temperatureList[i] = T_
    #         T_ *= alpha
    #     end


    
    # elseif scheme == "scheme5"
    #     # alpha = (Tmin / Tmax) ^ (1 / steps)
    #     alpha = (Tmin / Tmax) ^ (1 / totalMoves)
    #     T__ = Tmax
    #     #
    #     ko = 0.9
    #     T_ = Tmax
    #     for i in 1:steps
    #         # newAlpha = ( (alpha) ^ i ) ^ (1 / tempLength) # rate between Tprevious and Tnext is alpha!
    #         for j in 1:tempLength
    #             k = ( (i - 1) * tempLength ) + j
    #             temperatureList[k] = T_
    #             T_ *= alpha
    #         end
    #         T_ /= alpha

    #         # ko = adjustK(T_, T__, ko, Tmax)
    #         # ko = alpha
    #         ko = T_ / T__
    #         T_ /= ko

    #         T__ *= alpha # this will be the esqueleton
    #     end

    
    
    end
    #
    # return temperatureList
end

# function adjustK(T_, T__, ko, Tmax)
#     temporal = T_
#     temporal /= ko
#     count = 0
#     ko = T_ / T__
#     # for i in 1:10
#     #     if temporal < T__
#     #         count = i
#     #         temporal *= ko
#     #         ko /= 1.1
#     #         temporal /= ko
#     #     # elseif temporal > 1.5 * T__
#     #     elseif temporal > Tmax
#     #         count = i
#     #         temporal *= ko
#     #         ko += 0.1
#     #         temporal /= ko
#     #     end
        
#     # end
#     println("count , ko: ", count , "  ",ko)
#     return ko
# end


##
##
################################################################################
