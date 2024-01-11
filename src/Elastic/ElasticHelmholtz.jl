module ElasticHelmholtz

using LinearAlgebra
using SparseArrays
using Helmholtz;
using jInv.Mesh;
using Multigrid;

using jInv.LinearSolvers
import jInv.Utils.clear!
import jInv.LinearSolvers.AbstractSolver
import jInv.LinearSolvers.solveLinearSystem

export ElasticHelmholtzParam

mutable struct ElasticHelmholtzParam
	Mesh   			:: RegularMesh;
	omega			:: Float64
	lambda          :: Union{Array{Float32},Array{Float64}};
	rho				:: Union{Array{Float32},Array{Float64}}
	mu				:: Union{Array{Float32},Array{Float64}}
	gamma  			:: Union{Array{Float32},Array{Float64}};
	NeumannOnTop	:: Bool
	MixedFormulation:: Bool
end


#include("elasticFaces.jl")
include("elasticFaces2.jl")
include("GetElasticHelmholtz.jl")

end


