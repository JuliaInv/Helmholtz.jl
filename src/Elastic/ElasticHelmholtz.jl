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
	lambda          :: Array{Float64};
	rho				:: Array{Float64}
	mu				:: Array{Float64}
	gamma  			:: Array{Float64};
	NeumannOnTop	:: Bool
	MixedFormulation:: Bool
end





#include("elasticFaces.jl")
include("GetElasticHelmholtz.jl")

end


