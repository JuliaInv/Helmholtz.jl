using jInv.Mesh;
using Helmholtz
using Helmholtz.ElasticHelmholtz
using Multigrid
using Multigrid.DomainDecomposition
using Multigrid.ParallelJuliaSolver

include("getModels.jl");

const TYPE = ComplexF64;
const ITYPE = Int64;

n = [256;128];
n_tup = tuple(n...);
dim = length(n);

pad    = 16;

(rho,lambda,mu,Minv) = getModel("linear", n);
q = getElasticPointSource(Minv,ComplexF64);

sp = sqrt.(rho[:]./(lambda[:] + 2*mu[:]))
ss = sqrt.(rho[:]./(mu[:]))
w = 0.97*getMaximalFrequency(ss.^2,Minv);

print("sp*omega*h = ",w*maximum(Minv.h)*maximum(sp),";    ");
println("ss*omega*h = ",w*maximum(Minv.h)*maximum(ss));
pad = pad*ones(Int64,3);
gamma = getCellCenteredABL(Minv,true,pad,w);
gamma .+= 0.01*w;

reformulated = true;
Hparam 		= ElasticHelmholtzParam(Minv,w,lambda,rho,mu,gamma,true,reformulated)
shift 		= 0.5;
ShiftT 		= GetElasticHelmholtzShiftOP(Hparam,shift)';
ShiftT       = convert(SparseMatrixCSC{TYPE,spIndType},ShiftT);

q = [q;zeros(eltype(q),prod(Minv.n))];

numCores 	= 2; 
maxIter     = 20;
relativeTol = 1e-6;
cycleType   ='K';
levels      = 3;
# relaxType   = "EconVankaFaces"; shift = 0.5; # here better to use shift = 0.5.
relaxType   = "VankaFaces";
relaxParam  = [0.5,0.3,0.3,0.2,0.0];
relaxParam = relaxParam[1:levels];
relaxPre 	= 1;
relaxPost   = 1;

MG = getMGparam(TYPE, ITYPE, levels,numCores,maxIter,relativeTol,relaxType,relaxParam, relaxPre,relaxPost,cycleType,"Julia",0.0,0.0,"SystemsFacesMixedLinear");

HrT = sparse(GetElasticHelmholtzOperator(Hparam)');
SHrT = HrT + ShiftT; #HrT = 0.0;
SHrT = convert(SparseMatrixCSC{TYPE,spIndType},SHrT);
H = (x) -> (SHrT'*x - ShiftT'*x);
x = copy(q); x[:] .= 0.0;

println("************************** RAP-based GEOMETRIC **************************************")

MG = MGsetup(SHrT,Minv,MG,1,true);
x,param,iter,resvec = solveGMRES_MG(H,MG,q,zeros(eltype(q),size(q)),true,5,true);
println("Took ",length(resvec)," iterations, rel error: ",norm(H(x)-q)/norm(q));

println("************************** Stencil-based GEOMETRIC **************************************")

Hparam_shifted = ElasticHelmholtzParam(Hparam.Mesh,Hparam.omega,Hparam.lambda,Hparam.rho,
						Hparam.mu,Hparam.gamma .+ shift*Hparam.omega, Hparam.NeumannOnTop,Hparam.MixedFormulation);

function restrictParam(mesh_fine,mesh_coarse,param_fine,level) 
	gamma_c  = restrictCellCenteredVariables(param_fine.gamma,mesh_fine.n);
	mu_c 	 = restrictCellCenteredVariables(param_fine.mu,mesh_fine.n);
	lambda_c = restrictCellCenteredVariables(param_fine.lambda,mesh_fine.n);
	rho_c 	 = restrictCellCenteredVariables(param_fine.rho,mesh_fine.n);
	return ElasticHelmholtzParam(mesh_coarse,param_fine.omega,lambda_c,rho_c,mu_c,gamma_c,param_fine.NeumannOnTop,param_fine.MixedFormulation);
end
getOperator(mesh,Hparam) = GetElasticHelmholtzOperator(Hparam); 
MGsetup(getMultilevelOperatorConstructor(Hparam_shifted,getOperator,restrictParam),Hparam_shifted.Mesh,MG,1,true);
x = copy(q); x[:] .= 0.0;
x,param,iter,resvec = solveGMRES_MG(H,MG,q,zeros(eltype(q),size(q)),true,5,true)
println("Took ",length(resvec)," iterations, rel error: ",norm(H(x)-q)/norm(q));