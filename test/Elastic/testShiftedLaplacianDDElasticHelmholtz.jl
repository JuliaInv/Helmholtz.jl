using jInv.Mesh;
using Helmholtz
using Helmholtz.ElasticHelmholtz
using Multigrid
using Multigrid.DomainDecomposition
using Multigrid.ParallelJuliaSolver

include("getModels.jl");

const TYPE = ComplexF64;
const ITYPE = Int64;

n = [128;64];
n_tup = tuple(n...);
dim = length(n);

pad    = 16;

(rho,lambda,mu,Minv) = getModel("linear", n);
q = getElasticPointSource(Minv,ComplexF64);

sp = sqrt.(rho[:]./(lambda[:] + 2*mu[:]))
ss = sqrt.(rho[:]./(mu[:]))
w = getMaximalFrequency(ss.^2,Minv);

print("sp*omega*h = ",w*maximum(Minv.h)*maximum(sp),";    ");
println("ss*omega*h = ",w*maximum(Minv.h)*maximum(ss));
pad = pad*ones(Int64,3);
gamma = getCellCenteredABL(Minv,true,pad,w);
gamma .+= 0.0001*w;

reformulated = false;
Hparam 		= ElasticHelmholtzParam(Minv,w,lambda,rho,mu,gamma,true,reformulated)
shift 		= 0.5;
Shift 		= GetElasticHelmholtzShiftOP(Hparam,shift);
Shift       = convert(SparseMatrixCSC{TYPE,spIndType},Shift);


numCores 	= 8; 
maxIter     = 50;
relativeTol = 1e-6;


HrT = sparse(GetElasticHelmholtzOperator(Hparam)');

if Hparam.MixedFormulation
	q = [q;zeros(eltype(q),prod(Minv.n))];
	DDparam = getDomainDecompositionParam(Minv,NumCells,overlap,getFacesStaggeredIndicesOfCell,getParallelJuliaSolver(TYPE,ITYPE,numCores=2,backend=3));
else
	DDparam = getDomainDecompositionParam(Minv,NumCells,overlap,getFacesStaggeredIndicesOfCellNoPressure,getParallelJuliaSolver(TYPE,ITYPE,numCores=2,backend=3));
end
SHrT = HrT + Shift;#HrT = 0.0;Shift = 0.0;
SHrT = convert(SparseMatrixCSC{TYPE,spIndType},SHrT);

x = copy(q); x[:] .= 0.0;

(~,DDparam) = solveLinearSystem!(SHrT,[],[],DDparam)
x = solveLinearSystem!(SHrT,q,x,DDparam)[1];
println(norm(SHrT'*x - q)/norm(q))


x = 0.0;