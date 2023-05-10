using jInv.Mesh;
using Helmholtz
using Helmholtz.ElasticHelmholtz
using Multigrid
using Multigrid.DomainDecomposition
using Multigrid.ParallelJuliaSolver

include("getModels.jl");
include("utils.jl")

const TYPE = ComplexF64;
const ITYPE = Int64;

n = [512;256];
shift = 0.25;
reformulated = false;
(Minv,Hparam,q,HrT,SHT,w) = getElasticHelmholtzProblem(;n=n,w_factor=0.9,model="linear",rho = 1.0,lambda = 1.0,mu = 1.0,
														pad=16,TYPE = ComplexF64,reformulated = reformulated,shift=shift);


NumCells = [4,2];
overlap = [8,8];


if Hparam.MixedFormulation
	SHT = blockdiag(SHT,spzeros(prod(Minv.n),prod(Minv.n)))';
	q = [q;zeros(eltype(q),prod(Minv.n))];
	DDparam = getDomainDecompositionParam(Minv,NumCells,overlap,getFacesStaggeredIndicesOfCell,getParallelJuliaSolver(TYPE,ITYPE,numCores=4,backend=3));
else
	DDparam = getDomainDecompositionParam(Minv,NumCells,overlap,getFacesStaggeredIndicesOfCellNoPressure,getParallelJuliaSolver(TYPE,ITYPE,numCores=4,backend=3));
end
SHrT = HrT + SHT;

x = copy(q); 
x[:] .= 0.0;

# println("Performing Dirichlet Setup") # the following line just calls for the setup:
# println("NumCells = ",NumCells," overlapp = ",overlap);
# (~,DDparam) = solveLinearSystem!(SHrT,[],[],DDparam); 


# println("Performing Dirichlet Solution with GMRES");
# x = solveLinearSystem!(HrT,q,x,DDparam)[1];

# println("Outside error: ", norm(HrT'*x - q)/norm(q));

# DDparam = setupDDSerial(SHrT,DDparam);
# x, = solveDDSerial(SHrT,q,zeros(ComplexF64,size(q)),DDparam,20);
# println("Outside error: ", norm(SHrT'*x - q)/norm(q))
x[:] .= 0.0;


###########################################################################
###########################################################################


println("Performing Absorbing+Neumann Setup")
println("NumCells = ",NumCells," overlapp = ",overlap);

function getSubParams(Hparam, M::RegularMesh,i::Array{Int64},NumCells::Array{Int64},Overlap::Array{Int64})
	subMesh   = getSubMeshOfCell(NumCells,overlap,i,M);
	IIp       = getCellCenteredIndicesOfCell(NumCells,overlap,i,M.n);
	code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2]];
	subgamma  = getABL(subMesh.n,i[end]==1,overlap.+8,2.0./(M.h[1]),code);#.+0.005*Hparam.omega;
	Hparam    = ElasticHelmholtzParam(subMesh,Hparam.omega,Hparam.lambda[IIp],Hparam.rho[IIp],Hparam.mu[IIp],Hparam.gamma[IIp]+subgamma[:],false,Hparam.MixedFormulation);
	return Hparam;
end


getDDMass = (ddp,hp,i)->(0.0.*Vector(diag(GetElasticHelmholtzShiftOP(hp,shift))));
Ctor = DomainDecompositionOperatorConstructor(Hparam,getSubParams,GetElasticHelmholtzOperator,getDDMass);
DDparam = setupDDSerial(Ctor,DDparam);
println("Performing DD Solution with GMRES")
x[:] .= 0.0;
x = solveLinearSystem!(HrT,q,x,DDparam)[1];
println("Outside error: ", norm(HrT'*x - q)/norm(q))

