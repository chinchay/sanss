# using TimerOutputs # for `to`
## Create a TimerOutput, this is the main type that keeps track of everything.
# const to = TimerOutput()

# using DelimitedFiles # for writedlm()

include("./customreadlammps.jl")
using .readLammpsModule
# using PyPlot
#
# using StatsBase
# using Statistics # for `std`
### \notit for `not in` : https://stackoverflow.com/questions/59978282/is-there-an-elegant-way-to-do-not-in-in-julia

# using Random # for `randperm`
using Test # for @test
# using Plots

include("./common.jl")
include("./utils.jl")

##
##
################################################################################

meanEnergies = []
meanStds = []
listSteps = [10] #[100_000] #[200] #[400]#[1600] #50 # 32000 40000
runs = 1 #10
tempLength = 10_000 # 1 #100_000 #100_000 #100 #50 #100 ###stepsTconstant
scheme = "linear" # options: "linear", "constant" # for tempLength
nWalkers = 1 #50 #20 #20
nCfgNeighbors = 1 #
walkersBornEqual = false
followAlgorithm1 = false #true # `true` for the original algorithm (for simulation). `false` for ""
# alpha = 0.9 <<<--- I will define later as = (Tmin/Tmax) ^ (1/steps)

##
# Base algorithm:
L1 = 2  #1 #2
L2 = 50 #50 #0 #50
Nv = 72 #1 #72
L  = L1 + L2
Ne = Nv - L # Number of empty vacancies
cutOff = 10.0 #4.3 #10.0 #4.3 #6.0
Tmax = 350 #200 #1000 #3000 #1000
Tmin = 1   #1    #10
file = "testVac/data.lammps"
Tmax = float(Tmax)
# Tfactor = -log(Tmax / Tmin)
# distSite2Col, neighbors_of, charges, ion1, ion2, removedSites,
#     U, UionAionBdic, UionAionBlist = initialize(L1, L2, Nv, cutOff, file)
distSite2Col, neighbors_of, charges, ion1, ion2, removedSites,
                    U, UionAionBlist = initialize(L1, L2, Nv, cutOff, file)
#
L_list  = [ i for i in 1:L ]
Ne_list = [ i for i in 1:Ne ]

##
################################################################################
energyBase = getEnergyBase(removedSites, charges, distSite2Col, neighbors_of)
println("energyBase: ", energyBase)
##
##
################################################################################
################################################################################
##
# DETERMINE Tmin AND Tmax from plot?? approximate, sure
# see plot `mygraph.png`:
# plotTmaxTminFromSamples( L1, L, Nv, ion1, ion2, removedSites, U,
#                         UionAionBlist, L_list, Ne_list )
##
################################################################################

# EQUILIBRATION STAGE
# free number of steps, stop according to the acceptances

followAlgorithm1 = false
#
steps = listSteps[1]
alpha = (Tmin / Tmax) ^ (1 / steps)
println("moves to perform: ", steps)
println("Tmax, Tmin, alpha: ", Tmax, " | ", Tmin, " | ", alpha)
# T_list = [ [ Tmax * exp(Tfactor * s / steps) for s in 1:steps ] for steps in listSteps ]
# T_list = [ Tmax * exp(Tfactor * s / steps) for s in 1:steps ]
#
@time begin
w_r, w_n, w_E, w_acc, record_T = equilibration( L1, L, Nv, ion1, ion2, removedSites,
                        UionAionBlist, nWalkers, walkersBornEqual, Tmax,
                        tempLength, scheme, steps, alpha )
#
end
println("Chekcing energies:")
for w in 1:nWalkers
    energy = getWalkerTotalEnergy( energyBase, w_r[w], U, UionAionBlist )
    println( "energy of opt: ", energy )
end
# w_r, w_n, w_E, w_acc, record_T = simulAnneal( L1, L, Nv, ion1, ion2, removedSites,
#                         UionAionBlist, nWalkers, walkersBornEqual, Tmax,
#                         tempLength, scheme, steps, alpha, followAlgorithm1 )
println("finished equilibration")

##
factNumPoints= 10
plotWalkers( factNumPoints, w_E, record_T )
# saving for future use:
w_r_eq = deepcopy(w_r)
w_n_eq = deepcopy(w_n)
w_E_eq = deepcopy(w_E)
w_acc_eq = deepcopy(w_acc)
record_T_eq = deepcopy(record_T)

#
##
throw(DivideError())

##

# SIMULATION STAGE

# Tmax = record_T_eq[end-3000]
# Tmax = record_T_eq[end-1000]
# Tmax = record_T_eq[end-500]
# Tmax = record_T_eq[end-10_00]
Tmax = record_T_eq[end-100]
# Tmax = record_T_eq[end-50]
# Tmax = record_T_eq[end-20]
followAlgorithm1 = true
#
steps = listSteps[1]
alpha = (Tmin / Tmax) ^ (1 / steps)
println("****** SIMULATION STAGE ******")
println("moves to perform: ", steps)
println("Tmax, Tmin, alpha: ", Tmax, " | ", Tmin, " | ", alpha)
#
@time begin
# loading:
w_r = deepcopy(w_r_eq)
w_n = deepcopy(w_n_eq)
#
w_r, w_n, w_E, w_acc, record_T = simulation!( w_r, w_n,
                        L1, L, Nv, ion1, ion2, removedSites,
                        UionAionBlist, nWalkers, walkersBornEqual, Tmax,
                        tempLength, scheme, steps, alpha )
#
end
#
println("=========")
println(w_r[1])
println("energy: ", w_E[1][end])

# w_r, w_n, w_E, w_acc, record_T = simulAnneal( L1, L, Nv, ion1, ion2, removedSites,
#                         UionAionBlist, nWalkers, walkersBornEqual, Tmax,
#                         tempLength, scheme, steps, alpha, followAlgorithm1 )
#
println("finished simulation stage.")

##
factNumPoints= 10
plotWalkers( factNumPoints, w_E, record_T )

##
# ORIGINAL SCHEME: RURS: RANDOM UPDATE OF A RANDOM SITE *************************
#
followAlgorithm1 = true
#
steps = listSteps[1]
alpha = (Tmin / Tmax) ^ (1 / steps)
println("****** RURS: RANDOM UPDATE OF A RANDOM SITE ******")
println("STEPS: ", steps)
println("TEMPLENGTH: ", tempLength)
println("Tmax, Tmin, alpha: ", Tmax, " | ", Tmin, " | ", alpha)
#
@time begin
w_r, w_n, w_E, w_acc, record_T = simulAnneal( L1, L, Nv, ion1, ion2, removedSites,
                        UionAionBlist, nWalkers, walkersBornEqual, Tmax,
                        tempLength, scheme, steps, alpha, followAlgorithm1 )
end
println("=========")
println(w_r[1])
println("energy: ", w_E[1][end])
println("finished RURS.")
#
##
factNumPoints= 10
plotWalkers( factNumPoints, w_E, record_T )


##

# ORIGINAL SCHEME: RURS: RANDOM UPDATE OF A RANDOM SITE *************************
# SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION
followAlgorithm1 = true
#
nRepeats = 5
steps = listSteps[1]
alpha = (Tmin / Tmax) ^ (1 / steps)
println("****** RURS: RANDOM UPDATE OF A RANDOM SITE ******")
println("****** SEVERAL RUNINGS BEGINING WITH THE SAME CONFIGURATION ********")
println("STEPS: ", steps)
println("TEMPLENGTH: ", tempLength)
println("Tmax, Tmin, alpha: ", Tmax, " | ", Tmin, " | ", alpha)
#
y     = []
lE   = [ zeros(0) for w in 1:nRepeats]
@time begin
    w_r0, w_n0, _, _, _, _, _, _, _ =
        getInitWalkers(L1, L, Nv, ion1, ion2, removedSites,
            UionAionBlist, nWalkers, walkersBornEqual, energyBase)
    #
    for i in 1:nRepeats
        println("i: ", i)
        # loading:
        w_r = deepcopy(w_r0)
        w_n = deepcopy(w_n0)
        #
        w_r, w_n, w_E, w_acc, record_T = simulation!( w_r, w_n,
                            L1, L, Nv, ion1, ion2, removedSites,
                            UionAionBlist, nWalkers, walkersBornEqual, Tmax,
                            tempLength, scheme, steps, alpha )
        #
        lE[i] = w_E[1]
        println("energy: ", w_E[1][end])
    end
end
println("=========")
# println(w_r[1])
# println("energy: ", w_E[1][end])
println("finished RURS beginning with the same configuration.")
#
##
factNumPoints= 50
# plotWalkers( factNumPoints, w_E, record_T )
plotWalkers( factNumPoints, lE, record_T )



##
# using Profile
# @profiler simulAnneal( L1, L, Nv, ion1, ion2, removedSites,
#                 UionAionBlist, nWalkers, walkersBornEqual, Tmax,
#                 tempLength, scheme, steps, alpha, followAlgorithm1 )
# print("fin profile.")
##
factNumPoints= 10
plotWalkers( factNumPoints, w_E, record_T )

##
for w in 1:nWalkers
    println(w_E[w][end])
end
##

if nWalkers > 2
    endings = [ w_E[w][end] for w in 1:nWalkers ]
    println(  "min, mean, std: ", minimum(endings),
                            "  ", mean(endings),
                            "  ", std(endings) ) # for std `using Statistics`
end

##
println("=========")
println(w_r[1])
println("energy: ", w_E[1][end])

##
################################################################################

##
nRepeats = 5
listSteps = [5] #[100_000] #[200] #[400]#[1600] #50 # 32000 40000
tempLength = 20000 # 1 #100_000 #100_000 #100 #50 #100 ###stepsTconstant
scheme = "linear" # options: "linear", "constant" # for tempLength
Tmax = 350 #200 #1000 #3000 #1000
Tmin = 1   #1    #10
Tmax = float(Tmax)
steps = listSteps[1]
alpha =  (Tmin / Tmax) ^ (1 / steps)
lE, record_T = experiment10( nRepeats, L1, L, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, walkersBornEqual, Tmax, tempLength, scheme, steps, alpha )
print("finished.")

#
##
neighbors_EQ = 1 #10 #50
nCheck_EQ = 100
nRepeats = 5
listSteps = [1000] #[100_000] #[200] #[400]#[1600] #50 # 32000 40000
tempLength = 100 # 1 #100_000 #100_000 #100 #50 #100 ###stepsTconstant
scheme = "linear" # options: "linear", "constant" # for tempLength
Tmax = 350 #350 #1400 #700 #350 #200 #1000 #3000 #1000
Tmin = 1   #1    #10
Tmax = float(Tmax)
steps = listSteps[1]
alpha =  (Tmin / Tmax) ^ (1 / steps)

walkersBornEqual = false
# # @time lE, record_T = experiment11( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, walkersBornEqual, Tmax, tempLength, scheme, steps, alpha )
# @time experiment11( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, walkersBornEqual, Tmax, tempLength, scheme, steps, alpha, nCfgNeighbors )
# print("finished.")

# experiment12 : only RURS
# @time experiment12( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, walkersBornEqual, Tmax, tempLength, scheme, steps, alpha, nCfgNeighbors )
# print("finished.")

# in this experiment13, nCfgNeighbors are chosen according to T, see `getNeighborsToScan()`. Also I have removed `walkersBornEqual` that I have not using at all.
# @time experiment13( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ )

# this experiment will use the `refill` from the previous loop, `nRepeats` times
displayMessages = false
@time experiment14( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )

# only RURS
# @time contador = experiment15( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
print("finished.")

# using Profile
# @profiler experiment11( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, walkersBornEqual, Tmax, tempLength, scheme, steps, alpha )

# answer = string(contador)
# run(`sayProgramFinished $answer`)   
##
##
###########
neighbors_EQ = 1 #10 #50
nCheck_EQ = 100
nRepeats = 1 #5 # 100
listSteps = [10] #[100_000] #[200] #[400]#[1600] #50 # 32000 40000
tempLength = 10000 # 1 #100_000 #100_000 #100 #50 #100 ###stepsTconstant
scheme = "linear" # options: "linear", "constant" # for tempLength
# scheme = "constant" # options: "linear", "constant" # for tempLength
# Tmax = 350 #350 #1400 #700 #350 #200 #1000 #3000 #1000
Tmax = 800
# Tmax = 2.667
Tmin = 1   #1    #10
Tmax = float(Tmax)
steps = listSteps[1]
alpha =  (Tmin / Tmax) ^ (1 / steps)
# alpha =  (Tmin / 800.0) ^ (1 / steps)  ## @@@@@@@@ <<<< AVISA QUE DEBISTE HABER USADO ESTE ALPHA @@@@@@@@@@@@@

walkersBornEqual = false
displayMessages = false

for i in 1:1
# only RURS
# @time contador = experiment15( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )

# EQ+RURS
@time contador = experiment16( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
print("finished.")
answer = string(contador)
run(`sayProgramFinished $answer`)   
end


#

# listSteps = [10] #[100_000] #[200] #[400]#[1600] #50 # 32000 40000
# tempLength = 100 # 1 #100_000 #100_000 #100 #50 #100 ###stepsTconstant
# Tmax = 800.0
# Tmin = 1.0
# Tmax = float(Tmax)
# Tmin = float(Tmin)
# steps = listSteps[1]
# #
# totalMoves = steps * tempLength
# temperatureList = zeros(totalMoves)

# fig, ax1 = PyPlot.subplots()
# x = collect(1:totalMoves)
# myColors = ["k", "r", "g", "b", "y"]

# # scheme = "scheme1"
# # getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )
# # # ax1.scatter(x, temperatureList, s=1.0, color=myColors[1], marker="o" )
# # ax1.plot(x, temperatureList, color=myColors[1], linestyle="-" )

# scheme = "scheme2"
# getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )
# # ax1.scatter(x, temperatureList, s=10.0, color=myColors[2], marker="+" )
# ax1.plot(x, temperatureList, color=myColors[2], linestyle="-" )

# # scheme = "scheme3"
# # getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )
# # # ax1.scatter(x, temperatureList, s=10.0, color=myColors[3], marker="x" )
# # ax1.plot(x, temperatureList, color=myColors[3], linestyle="-" )


# # scheme = "scheme4"
# # getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )
# # # ax1.scatter(x, temperatureList, s=10.0, color=myColors[4], marker="|" )
# # ax1.plot(x, temperatureList, color=myColors[4], linestyle="-" )

# scheme = "scheme5"
# getTemperatureList!( Tmax, Tmin, totalMoves, steps, tempLength, temperatureList, scheme )
# # ax1.scatter(x, temperatureList, s=10.0, color=myColors[3], marker="x" )
# ax1.plot(x, temperatureList, color=myColors[5], linestyle="-" )


# ax1.set_ylabel("effective temperature", color="k")
# fig.savefig("effective_temperature.png")
# PyPlot.close(fig)


# ##

# # function getAcceptanceRate()

# # end

##

##
##
###########
neighbors_EQ = 1 #10 #50
nCheck_EQ = 100
nRepeats = 1 #5 # 100
listSteps = [100] #[100_000] #[200] #[400]#[1600] #50 # 32000 40000
tempLength = 100 # 1 #100_000 #100_000 #100 #50 #100 ###stepsTconstant
# scheme = "linear" # options: "linear", "constant" # for tempLength
scheme = "scheme6"
# scheme = "constant" # options: "linear", "constant" # for tempLength
# Tmax = 350 #350 #1400 #700 #350 #200 #1000 #3000 #1000
Tmax = 800
# Tmax = 2.667
Tmin = 1   #1    #10
Tmax = float(Tmax)
steps = listSteps[1]
alpha =  (Tmin / Tmax) ^ (1 / steps)
# alpha =  (Tmin / 800.0) ^ (1 / steps)  ## @@@@@@@@ <<<< AVISA QUE DEBISTE HABER USADO ESTE ALPHA @@@@@@@@@@@@@

walkersBornEqual = false
displayMessages = false

# contador = 0
# global n = 1
# while ( contador == 0 ) && ( n <= 50 )
#     global n += 1
#     # EQ+RURS
#     @time contador = experiment17( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
# end

for i in 1:1
    # only RURS
    # @time contador = experiment15( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )

    # EQ+RURS
    @time contador = experiment17( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
end


# print("finished.")
# answer = string(contador)
# run(`sayProgramFinished $answer`)


#
##

##
###########
neighbors_EQ = 1 #10 #50
nCheck_EQ = 100
nRepeats = 1 #5 # 100
listSteps = [100] #[100_000] #[200] #[400]#[1600] #50 # 32000 40000
tempLength = 100 # 1 #100_000 #100_000 #100 #50 #100 ###stepsTconstant
# scheme = "linear" # options: "linear", "constant" # for tempLength
scheme = "scheme1"
# scheme = "constant" # options: "linear", "constant" # for tempLength
# Tmax = 350 #350 #1400 #700 #350 #200 #1000 #3000 #1000
Tmax = 800
# Tmax = 2.667
Tmin = 1   #1    #10
Tmax = float(Tmax)
steps = listSteps[1]
alpha =  (Tmin / Tmax) ^ (1 / steps)
# alpha =  (Tmin / 800.0) ^ (1 / steps)  ## @@@@@@@@ <<<< AVISA QUE DEBISTE HABER USADO ESTE ALPHA @@@@@@@@@@@@@

walkersBornEqual = false
displayMessages = false

# contador = 0
# global n = 1
# while ( contador == 0 ) && ( n <= 50 )
#     global n += 1
#     # EQ+RURS
#     @time contador = experiment17( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
# end

for i in 1:1
    # only RURS
    # @time contador = experiment15( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )

    # EQ+RURS
    @time contador = experiment18( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
end


# print("finished.")
# answer = string(contador)
# run(`sayProgramFinished $answer`)


#
####################
##
# Wednesday April 7th, 2021:
# I forgot how the code was structures. I need to rearrange the code. So far,
# experiment18() seems to be the best algorithm (although not aproved because "not stable") to find the 
# ground state configuration. It uses `RURS_3_scheme1()` in util.jl, which uses 
# getNeighborsToScan() which is key to find the GS fast. I am going to copy the previous lines 
# so to avoid modifying the code.
# ##
###########
neighbors_EQ = 1 #10 #50
nCheck_EQ = 100
nRepeats = 2 #5 # 100
listSteps = [1000] #[100_000] #[200] #[400]#[1600] #50 # 32000 40000
tempLength = 100 # 1 #100_000 #100_000 #100 #50 #100 ###stepsTconstant
# scheme = "linear" # options: "linear", "constant" # for tempLength
scheme = "scheme1"
# scheme = "constant" # options: "linear", "constant" # for tempLength
# Tmax = 350 #350 #1400 #700 #350 #200 #1000 #3000 #1000
Tmax = 800
# Tmax = 2.667
Tmin = 1   #1    #10
Tmax = float(Tmax)
steps = listSteps[1]
alpha =  (Tmin / Tmax) ^ (1 / steps)
# alpha =  (Tmin / 800.0) ^ (1 / steps)  ## @@@@@@@@ <<<< AVISA QUE DEBISTE HABER USADO ESTE ALPHA @@@@@@@@@@@@@

walkersBornEqual = false
displayMessages = false

# contador = 0
# global n = 1
# while ( contador == 0 ) && ( n <= 50 )
#     global n += 1
#     # EQ+RURS
#     @time contador = experiment17( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
# end

for i in 1:1
    # only RURS
    @time contador = experiment15( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
    
    # EQ+RURS
    # @time contador = experiment18( nRepeats, L1, L, Ne, Nv, ion1, ion2, removedSites, UionAionBlist, nWalkers, Tmax, tempLength, scheme, steps, alpha, neighbors_EQ, nCheck_EQ, displayMessages )
end


# print("finished.")
# answer = string(contador)
# run(`sayProgramFinished $answer`)


#

