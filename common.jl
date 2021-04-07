# using TimerOutputs # for `to`
## Create a TimerOutput, this is the main type that keeps track of everything.
# const to = TimerOutput()

using DelimitedFiles # for writedlm()

include("./myreadlammps.jl")
using .readLammpsModule
using PyPlot
#
using StatsBase
using Statistics # for `std`
### \notit for `not in` : https://stackoverflow.com/questions/59978282/is-there-an-elegant-way-to-do-not-in-in-julia

using Random # for `randperm`
using Test # for @test

function energyInteraction(
                dist::Float64,
                QA::Float64,
                QB::Float64,
                cutOff::Float64)
    # according to https://lammps.sandia.gov/doc/dielectric.html
    # the default dielectric value, "epsilon", is 1.0
    # we are assuming using pair_style coul/cut
    # C is an energy-conversion constant: (see https://lammps.sandia.gov/doc/pair_coul.html)
    # you need to find the value of C:
    # C = 14.399645000 #1
    # if dist <= cutOff:
    #     e = COEFF1 * QA * QB / dist # coulomb
    # else:
    #     e = 0
    # #
    # return e
    # return 1
    # we dont't need to check cutoff, neighbors were already checked, so:
    return 14.399645000 * QA * QB / dist
end

function getStructure(originalFile::String)
    # include("./myreadlammps.jl")
    # using .readLammpsModule
    ##
    # originalFile = "testVac/data.lammps" # file without vacancies
    structure, charges_ = readLammpsModule.getPymatgenStructFromLammpsInput(originalFile, "pymatgen")
    # convert charges_ pointer to python object to Julia array:
    n = length(charges_)
    charges = zeros(n)
    for i in 1:n
        charges[i] = charges_[i]
    end
    ############
    # import pymatgen as mg
    # lattice = mg.Lattice.cubic(4.2)
    # # structure = mg.Structure(lattice, ["Cs", "Cl"],
    # #                 [ [0, 0, 0], [0.5, 0.5, 0.5]] )

    # structure = mg.Structure(lattice, ["La"],
    #                 [ [0, 0, 0], ] )
    # structure.make_supercell(2)
    # structure[0] = "Li", [0, 0, 0]
    # charges = getCharges(structure)
    ############


    #
    return structure, charges
end

# function whereInt(array, elementToFind)
function whereInt(
            array::Array{Int,1},
            elementToFind::Int
            )
    #
    numberOfCoincidences = 0
    n = length(array)
    for i in 1:n
        e = array[i]
        if e == elementToFind
            numberOfCoincidences += 1
        end
    end
    #
    if numberOfCoincidences == 0
        return zeros(Int, 0), 0
    else
        listOfIndexes = zeros(Int, numberOfCoincidences)
        j = 0
        for i in 1:n
            e = array[i]
            if e == elementToFind
                j += 1
                listOfIndexes[j] = i
            end
        end
        return listOfIndexes, numberOfCoincidences
    end

    # listOfIndexes = []
    # numberOfCoincidences = 0
    # for (iterator, element) in enumerate(array)
    #     if element == elementToFind
    #         numberOfCoincidences += 1
    #         append!(listOfIndexes, iterator)
    #     end
    # end
    # return listOfIndexes, numberOfCoincidences
end

# function getDistNeighbors(structure::PyCall.PyObject, cutOff::Float64)
function getDistNeighbors(structure, cutOff::Float64)
    # https://pymatgen.org/pymatgen.core.structure.html
    sites_, neighbors_, _, distances = structure.get_neighbor_list(cutOff)
    #
    # fixing indices to be compatible with Julia beginning in 1 !!!
    # for i in eachindex(sites)
    nn = length(sites_) # this is NOT the number of sites, since here they are REPEATED!!!
    sites     = zeros(Int, nn) # Integer
    neighbors = zeros(Int, nn) # Integer
    for i in 1:nn
        # sites_[i] += 1
        sites[i] = sites_[i] + 1

        # neighbors_[i] += 1
        neighbors[i] = neighbors_[i] + 1
    end
    #
    num_sites = length(structure.cart_coords)
    neighbors_of = [ zeros(Int, 0) for s in 1:num_sites ] # array of arrays
    distSite2Col = [ zeros(0)      for s in 1:num_sites ]
    # for s in range(len(structure.cart_coords)):  Python
    # for s in eachindex(structure.cart_coords)
    for s in 1:num_sites
        # array `sites` is a numpy array, so the np.where can work:
        # columns  = np.where(sites == s)[0] # python
        columns, nCol   = whereInt(sites, s)
        neighbors_of[s] = zeros(Int, nCol) # Integer
        distSite2Col[s] = zeros(nCol) # Float
        i = 0
        for col in columns
            i += 1
            neighbors_of[s][i] = neighbors[col]
            distSite2Col[s][i] = distances[col]
        end
    end
    return distSite2Col, neighbors_of
end

function getRemovedSites( charges::Array{Float64,1} )
    # Now we will remove all Lithium atoms (Nv Lithium atoms):
    Q_Li  = 1.0 # Z_li=3, but its charge in the input lammps file is 1.0000
    Nv = 0
    for q in charges
        if q == Q_Li
            Nv += 1
        end
    end
    #
    removedSites = zeros(Int, Nv)
    i = 0
    for (s, q) in enumerate(charges)
        if q == Q_Li
            i += 1
            removedSites[i] = s
        end
    end
    #
    return removedSites
end


function initialize(L1::Int,
                    L2::Int,
                    Nv::Int,
                    cutOff::Float64,
                    file::String)
    L  = L1 + L2
    @test L <= Nv
    #
    # cutOff = 4.3 #10.0 #4.3 #6.0
    structure, charges = getStructure(file)
    positions = structure.cart_coords #position of each site
    distSite2Col, neighbors_of = getDistNeighbors(structure, cutOff)
    # atomic_numbers, Zion1, Zion2 = getAtomicNumbers(structure)
    Qion1 = 3.0 # ion1: Ga
    Qion2 = 1.0 # ion2: Li
    ion1  = 1 # This is because we are in JULIA, which begins in 1!
    ion2  = 2

    nSites = length(positions)
    @test Nv <= nSites

    # Part II.b
    # removedSites = getRemovedSites(atomic_numbers)
    # println(typeof(charges))
    removedSites = getRemovedSites(charges)
    # print(removedSites)
    @test length(removedSites) == Nv

    # Part II.c
    # Now calculatate the potential energy around
    # a vacancy as IF an ion was going to occupy it:
    U = zeros(Nv, 2)
    for (i, site) in enumerate(removedSites)
        for (col, neighborSite) in enumerate(neighbors_of[site])
            # "calculate" rij- distance between ATOM (not vacancy!) 'j' and vacancy 'i'
            # ...so we are removing atoms located at `removedSites` sites. Avoid them now:
            if neighborSite ∉ removedSites
                distance = distSite2Col[site][col]
                Q_B = charges[neighborSite]
                U[i, ion1] += energyInteraction(distance, Qion1, Q_B, cutOff)
                U[i, ion2] += energyInteraction(distance, Qion2, Q_B, cutOff)
            end
        end
    end
    whereDic       = tableWhere(removedSites, neighbors_of)
    pairInteracDic = tablePairInterac(removedSites, distSite2Col, whereDic, Qion1, Qion2, ion1, ion2, cutOff)
    UionAionBlist  = tableUionAionB(removedSites, whereDic, pairInteracDic, ion1, ion2)
    #
    # return distSite2Col, neighbors_of, charges, Qion0, Qion1, ion0, ion1, removedSites, U, UionAionBdic
    # return distSite2Col, neighbors_of, charges, ion1, ion2, removedSites, U, UionAionBdic, UionAionBlist
    return distSite2Col, neighbors_of, charges, ion1, ion2, removedSites, U, UionAionBlist

end

# function getEnergyBase( removedSites::Array{Int,1},
#                         charges::Array{Float64,1},
#                         distSite2Col::Array{Any,1},
#                         neighbors_of::Array{Int,1}
#                         )
function getEnergyBase( removedSites::Array{Int,1},
                        charges,
                        distSite2Col,
                        neighbors_of
    )

    e = 0
    # num_sites = length(charges)
    for site in eachindex(charges)
        if site ∉ removedSites
            for (col, neighborSite) in enumerate(neighbors_of[site])
                if neighborSite ∉ removedSites
                    distance = distSite2Col[site][col]
                    QA = charges[site]
                    QB = charges[neighborSite]
                    e += energyInteraction(distance, QA, QB, cutOff)
                end
            end
        end
    end
    #
    e *= 0.5
    return e
end

# s = getEnergy_ion_atoms( L, refill, U)
function getEnergy_ion_atoms(
                L::Int,
                refill::Array{Int64,2},
                U::Array{Float64,2}
                )
    # Ut = 0
    # for vec in refill
    #     # Ut += U[ refill[i][1], refill[i][3] ] # Python
    #     Ut += U[ vec[2], vec[4] ]
    # end
    # return Ut
    # see "Generatros" in https://calculuswithjulia.github.io/precalc/ranges.html
    # return sum( U[ vec[2], vec[4] ] for vec in refill ) # no brackets []

    s = 0.0
    for l in 1:L
        r   = refill[2,l]
        ion = refill[4,l]
        s += U[r, ion]
    end
    return s
end

function get_refill(L1::Int,
                    L::Int,
                    Nv::Int,
                    ion1::Int,
                    ion2::Int,
                    removedSites::Array{Int64,1}
                    )
    # From those vacancies, we will fill out with L ions, chosen randomly:
    # L random integers from 0 to Nv-1 without repetition:

    # il = random.sample(range(Nv), L)
    il = sample( 1:Nv, L, replace=false ) # `using StatsBase` is needed
    # refill = [ [ c, r, removedSites[r], c <= L1 ? ion1 : ion2 ] for (c, r) in enumerate(il) ]
    #
    refill = zeros(Int, 4, L)
    for (l, r) in enumerate(il)
        refill[ 1, l] = l
        refill[ 2, l] = r
        refill[ 3, l] = removedSites[r]
        refill[ 4, l] = l <= L1 ? ion1 : ion2
    end
    return refill
end

function is_r_in_2Refill( r::Int, refill::Array{Int64,2}, L::Int)
    found = false
    for k in 1:L
        if r == refill[2, k]
            found = true
        end
    end
    return found
end

# notRefill = get_notRefill(refill, L, Ne, Nv)
function get_notRefill(refill::Array{Int64,2}, L::Int, Ne::Int, Nv::Int)
    notRefill = zeros(Int, Ne)
    j = 0
    # found = false
    for r in 1:Nv
        if !is_r_in_2Refill(r, refill, L)
            j += 1
            notRefill[j] = r
        end
    end
    return notRefill
end

# function tableWhere(removedSites::Array{Any,1}, neighbors_of::Array{Int,1})
function tableWhere(removedSites, neighbors_of)
    # dic = {}
    dic = Dict()
    for A in removedSites
        neighborsOfA = neighbors_of[A]
        for B in removedSites
            # columns   = np.where(neighborsOfA == B)[0]
            columns, _   = whereInt(neighborsOfA, B)
            dic[A, B] = columns
        end
    end
    return dic
end

function tablePairInterac(
                        removedSites::Array{Int,1},
                        distSite2Col,
                        whereDic::Dict{Any,Any},
                        Qion1::Float64,
                        Qion2::Float64,
                        ion1::Int,
                        ion2::Int,
                        cutOff::Float64
                        )

    dic = Dict()
    for A in removedSites
        dist_A_To_col = distSite2Col[A]
        for B in removedSites
            # columns   = whereDic[A, B]
            for colB in whereDic[A, B]
                e = energyInteraction(dist_A_To_col[colB], Qion1, Qion1, cutOff)
                dic[A, B, colB, ion1, ion1] = e
                #
                e = energyInteraction(dist_A_To_col[colB], Qion1, Qion2, cutOff)
                dic[A, B, colB, ion1, ion2] = e
                dic[A, B, colB, ion2, ion1] = e
                #
                e = energyInteraction(dist_A_To_col[colB], Qion2, Qion2, cutOff)
                dic[A, B, colB, ion2, ion2] = e
            end
        end
    end
    return dic
end

function tableUionAionB(
                    removedSites::Array{Int,1},
                    whereDic::Dict{Any,Any},
                    pairInteracDic::Dict{Any,Any},
                    ion1::Int,
                    ion2::Int
                    )
    # dic = Dict()
    # for A in removedSites
    #     for B in removedSites
    #         dic[A, B, ion1, ion1] = 0
    #         dic[A, B, ion1, ion2] = 0
    #         dic[A, B, ion2, ion2] = 0
    #         for colB in whereDic[A, B]
    #             dic[A, B, ion1, ion1] += pairInteracDic[ A, B, colB, ion1, ion1 ]
    #             dic[A, B, ion1, ion2] += pairInteracDic[ A, B, colB, ion1, ion2 ]
    #             dic[A, B, ion2, ion2] += pairInteracDic[ A, B, colB, ion2, ion2 ]
    #         end
    #         dic[A, B, ion2, ion1] = dic[A, B, ion1, ion2]
    #     end
    # end

    iones = [ion1, ion2]
    #
    n = length(removedSites)
    m = length(iones)
    matrix = zeros( n, n, m, m )

    for l in iones
        for k in iones
            for (j, B) in enumerate(removedSites)
                for (i, A) in enumerate(removedSites)
                    for colB in whereDic[A, B]
                        matrix[i,j,k,l] += pairInteracDic[ A, B, colB, k, l ]
                    end
                end
            end
        end
    end

    # return dic, arr
    return matrix
end

# Uion = getEnergy_ion_ion( L, refill, UionAionBlist)
function getEnergy_ion_ion(
        L::Int,
        refill::Array{Int64,2},
        UionAionBlist::Array{Float64,4}
        )
    Uion = 0.0
    # L = length(refill)
    # we are considering only vacancy-vacancy interaction now, as if they
    # were going to be occupied by ions:
    for i in 1:L
        # _, _, A, ionA = refill[i] # =[ i, r, removedSites[r], ion ], i belongs [0,L-1]
        # _, ia, _, ionA = refill[i] # =[ i, r, removedSites[r], ion ], i belongs [0,L-1]
        ia   = refill[2, i]
        ionA = refill[4, i]
        for j in i+1:L # `i+1` to avoid double counting.
            # _, _, B, ionB = refill[j]
            # _, ib, _, ionB = refill[j]
            ib   = refill[2, j]
            ionB = refill[4, j]
            # Uion += getEnergyAB_dict(A, B, ionA, ionB, UionAionBdic)
            # Uion += UionAionBdic[A, B, ionA, ionB]
            Uion += UionAionBlist[ia, ib, ionA, ionB] ####
        end
    end
    return Uion
end

# function getRandomInt(alist::Array{Any, 1})
function getRandomInt(alist)
    # return random.choice(alist)
    return rand(alist)
end

# myCopyRefill!(L, refill, r)
function myCopyRefill!(
            L::Int,
            refill::Array{Int64,2},
            r::Array{Int64,2}
            )
    for i in 1:L
        for j in 1:4
            r[j,i] = refill[j,i]
        end
    end
end

# myCopyNotRefill!(Ne, notRefill, n)
function myCopyNotRefill!(
            Ne::Int,
            notRefill::Array{Int64,1},
            n::Array{Int64,1}
            )
    #
    for i in 1:Ne
        n[i] = notRefill[i]
    end
end



# get_dE2!(L, Ne, refill, notRefill, removedSites, U, UionAionBlist, L_list, rTemp, nTemp, sumdUdistr )
function get_dE2!(L::Int,
                Ne::Int,
                refill::Array{Int64,2},
                notRefill::Array{Int64,1},
                removedSites::Array{Int,1},
                U::Array{Float64, 2},
                UionAionBlist::Array{Float64,4},
                L_list::Array{Int,1},
                rTemp::Array{Int64,2},
                nTemp::Array{Int64,1},
                p_dEsum::Array{Float64,1}
                )

    # @timeit to "a1" L_list__    = copy(L_list)
    # refill__    = deepcopy(refill)
    # myCopyRefill!( refill, rTemp)
    myCopyRefill!(L, refill, rTemp)

    # myCopyNotRefill!( notRefill, nTemp )
    myCopyNotRefill!(Ne, notRefill, nTemp)

    sumdUdistr = 0.0
    # you need `using Random` for `randperm()`
    # for v in randperm(L) # It'll give a random permutation list of [1,2,3,...,L]
    for v in shuffle!(L_list)
    # for i in 1:L
        # v = random.choice(L_list)
        # w = random.choice(Ne_list)
        # @timeit to "b1" v = getRandomInt(L_list__)
        # L_list__.remove(v) # remove the element
        # @timeit to "b2" filter!( X -> X ≠ v, L_list__)

        # @timeit to "b3" w = getRandomInt(Ne_list)
        w = rand(1:Ne)

        #
        # c, ia, A, ionA = refill__[v] #`ia` goes from 0 to Nv-1. #=[ i, r, atomsRemoved[r], ion ]
        # @timeit to "b4" c    = rTemp[v][1]
        # ia   = rTemp[v][2]
        # A    = rTemp[v][3]
        # ionA = rTemp[v][4]
        ia   = rTemp[2, v]
        A    = rTemp[3, v]
        ionA = rTemp[4, v]

        #
        # @test c == v
        #
        # ib = notRefill_new[w] # `ib` goes from 0 to Nv-1
        ib = nTemp[w] # `ib` goes from 0 to Nv-1
        B  = removedSites[ib]

        # IV.
        # Calculate diff
        # Calculate dUt first
        # @timeit to "b10" dUt = -U[ia, ionA] + U[ib, ionA]
        # @timeit to "b10a" temp1 = U[ia, ionA]
        # @timeit to "b10b" temp2 = U[ib, ionA]
        # @timeit to "b10c" dUt2  = temp2 - temp1
        # @timeit to "b10d" dUt   = -temp1 + temp2
        # @timeit to "b10e" dUt3 =  U[ib, ionA] - U[ia, ionA]
        dUt = -U[ia, ionA] + U[ib, ionA]

        # Ut += dUt

        # find dUion
        # UiA : energy between ion in ia and other ions in il
        # UiB : energy between ion in ib and other ions in il
        # dUion = UiB - UiA
        UiA = 0.0
        UiB = 0.0
        for l in 1:L
            k    = rTemp[1, l]
            ilk  = rTemp[2, l]
            ionX = rTemp[4, l]
            if k != v
                UiA += UionAionBlist[ia, ilk, ionA, ionX]
                UiB += UionAionBlist[ib, ilk, ionA, ionX]
            end
        end


        # for arr in rTemp
        #     # k, ilk, _, ionX = arr # =[ k, r, removedSites[r], ion ], k belongs [0;L-1]
        #     k    = arr[1]
        #     ilk  = arr[2]
        #     ionX = arr[4]
        #     # _, ilk, _, ionX = refill[k] # =[ k, r, removedSites[r], ion ], k belongs [0;L-1]
        #     if k != v
        #         UiA += UionAionBlist[ia, ilk, ionA, ionX]
        #         UiB += UionAionBlist[ib, ilk, ionA, ionX]
        #     end
        # end
        #
        # @timeit to "c*" for i in 1:L
        #     # # k, ilk, _, ionX = arr # =[ k, r, removedSites[r], ion ], k belongs [0;L-1]
        #     @timeit to "c1" k    = rTemp[i][1]
        #     @timeit to "c2" ilk  = rTemp[i][2]
        #     @timeit to "c3" ionX = rTemp[i][4]
        #     # _, ilk, _, ionX = refill[k] # =[ k, r, removedSites[r], ion ], k belongs [0;L-1]
        #     @timeit to "d0" if k != v
        #         @timeit to "d1" UiA += UionAionBlist[ia, ilk, ionA, ionX]
        #         @timeit to "d2" UiB += UionAionBlist[ib, ilk, ionA, ionX]
        #     end
        # end

        dUion   = UiB - UiA
        dUdistr = dUt + dUion # this is diff " delta(w, w') "
        #
        # refill__[v]    = [ v, ib, B, ionA ]
        # rTemp[v][1] = v
        # rTemp[v][2] = ib
        # rTemp[v][3] = B
        # rTemp[v][4] = ionA
        rTemp[1, v] = v
        rTemp[2, v] = ib
        rTemp[3, v] = B
        rTemp[4, v] = ionA

        #
        nTemp[w] = ia
        sumdUdistr += dUdistr
    end

    p_dEsum[1] = sumdUdistr # mutating its content
    #
    # return dUdistr, v, ib, B, ionA, w, ia
    # return sumdUdistr, rTemp, nTemp ######

end

# dUdistr, v, ib, B, ionA, w, ia = get_dE1(L, Ne, refill, notRefill, removedSites, U, UionAionBlist)
function get_dE1(
                L::Int,
                Ne::Int,
                refill::Array{Int64,2},
                notRefill::Array{Int64,1},
                removedSites::Array{Int,1},
                U::Array{Float64, 2},
                UionAionBlist::Array{Float64,4},
                )
    # v = getRandomInt(L_list)
    v = rand(1:L)
    # w = getRandomInt(Ne_list)
    w = rand(1:Ne)
    # v = rand(L_list)
    # w = rand(Ne_list)
    # v = random.choice(L_list)
    # w = random.choice(Ne_list)

    # c, ia, A, ionA = refill[v] #`ia` goes from 0 to Nv-1. #=[ i, r, atomsRemoved[r], ion ]
    # c    = refill[v][1]
    ia   = refill[2, v]
    # A    = refill[v][3]
    ionA = refill[4, v]
    # @test c == v
    #
    # ib = notRefill_new[w] # `ib` goes from 0 to Nv-1
    ib = notRefill[w] # `ib` goes from 0 to Nv-1
    B  = removedSites[ib]
    #
    # exchange ia and ib:
    # notRefill_new[w] = ia
    # refill_new[v] = [ v, ib, B, QA ]

    #
    # the new solution is w'= il(i:L)

    # IV.
    # Calculate diff
    # Calculate dUt first
    # dUt = -U[ia, ZA] + U[ib, ZA]
    dUt = -U[ia, ionA] + U[ib, ionA]
    # Ut += dUt

    # find dUion
    # UiA : energy between ion in ia and other ions in il
    # UiB : energy between ion in ib and other ions in il
    # dUion = UiB - UiA

    # UiA = sum( [ UionAionBlist[ia][ rx[1] ][ionA][ rx[3] ] for rx in refill ] )
    # UiB = sum( [ UionAionBlist[ib][ rx[1] ][ionA][ rx[3] ] for rx in refill if rx[0] != v ] )

    UiA = 0.0
    UiB = 0.0
    for l in 1:L
        k    = refill[1, l]
        ilk  = refill[2, l]
        ionX = refill[4, l]
        if k != v
            UiA += UionAionBlist[ia, ilk, ionA, ionX]
            UiB += UionAionBlist[ib, ilk, ionA, ionX]
        end
    end
    #
    # for arr in refill
    #     # k, ilk, _, ionX = arr # =[ k, r, removedSites[r], ion ], k belongs [0;L-1]
    #     k    = arr[1]
    #     ilk  = arr[2]
    #     ionX = arr[4]
    #     # _, ilk, _, ionX = refill[k] # =[ k, r, removedSites[r], ion ], k belongs [0;L-1]
    #     if k != v
    #         UiA += UionAionBlist[ia, ilk, ionA, ionX]
    #         UiB += UionAionBlist[ib, ilk, ionA, ionX]
    #     end
    # end
    #
    dUion = UiB - UiA

    # dUion = diff
    # Uion += dUion
    # Udistr = Ut + Uion
    dUdistr = dUt + dUion # this is diff " delta(w, w') "
    #
    return dUdistr, v, ib, B, ionA, w, ia
    # return Any[dUdistr, v, ib, B, ionA, w, ia]
end


function getBoltzmanFactor(dE::Float64, T::Float64) # flipEnergy, Temperature
    # kB = 1 # -> Then Temperature is not in Kelvin!
    # return math.exp( -dE / (kB * T) )
    return exp( -dE / T )
end

function belongs(x::Float64, xmin::Float64, xmax::Float64)
    return (xmin <= x) & (x <= xmax)
end
