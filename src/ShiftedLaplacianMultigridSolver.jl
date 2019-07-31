export ShiftedLaplacianMultigridSolver,updateParam,getShiftedLaplacianMultigridSolver,copySolver
export solveLinearSystem,clear!

mutable struct ShiftedLaplacianMultigridSolver <: AbstractSolver
	helmParam		:: HelmholtzParam
	MG				:: MGparam
	shift			:: Array{Float64}
	Krylov			:: String
	inner			:: Int64
	doClear			:: Int64 # flag to clear factorization
	verbose			:: Bool
	setupTime       ::Real
	nPrec           ::Int
	solveTime       ::Real
end

import jInv.LinearSolvers.copySolver;
function copySolver(s::ShiftedLaplacianMultigridSolver)
	# copies absolutely what's necessary.
	clear!(s.helmParam.Mesh);
	return getShiftedLaplacianMultigridSolver(s.helmParam,Multigrid.copySolver(s.MG),s.shift,s.Krylov,s.inner,s.verbose);
end

function getShiftedLaplacianMultigridSolver(helmParam::HelmholtzParam, MG::MGparam,shift::Array{Float64},Krylov::String="BiCGSTAB",inner::Int64=5,verbose::Bool = false)
	return ShiftedLaplacianMultigridSolver(helmParam,MG,shift,Krylov,inner,0,verbose,0.0,0,0.0);
end

function getShiftedLaplacianMultigridSolver(helmParam::HelmholtzParam, MG::MGparam,shift::Float64,Krylov::String="BiCGSTAB",inner::Int64=5,verbose::Bool = false)
	return getShiftedLaplacianMultigridSolver(helmParam,MG,ones(MG.levels)*shift,Krylov,inner,verbose);
end

function solveLinearSystem(ShiftedHT,B,param::ShiftedLaplacianMultigridSolver,doTranspose::Int64=0)
	if size(B,2) == 1
		B = vec(B);
	end
	if param.doClear==1
		clear!(param.MG);
	end
	if norm(B) == 0.0
		X = zeros(eltype(B),size(B));
		return X, param;
	end
	tt = time_ns()
	TYPE = eltype(B);
	n = size(B,1)
	nrhs = size(B,2)

	# build preconditioner
	if hierarchyExists(param.MG)==false
		## THIS CODE USES GEOMETRIC MULTIGRID
		# shift = param.shift;
		# Hparam_shifted = getShiftedHelmholtzParam(param.helmParam,shift[1]);
		# function restrictParam(mesh_fine,mesh_coarse,param_fine,level)
			# P = getShiftedHelmholtzParam(param_fine,shift[level+1] - shift[level]); 
			# P.gamma = restrictNodalVariables2(P.gamma,mesh_fine.n+1);
			# P.m = restrictNodalVariables2(P.m,mesh_fine.n+1); 
			# P.Mesh = mesh_coarse; 
			# return P;
		# end
		# getOperator(mesh,Hparam) =  GetHelmholtzOperator(mesh,Hparam.m,Hparam.omega,Hparam.gamma,Hparam.NeumannOnTop,Hparam.Sommerfeld);
		# MGsetup(getMultilevelOperatorConstructor(Hparam_shifted,getOperator,restrictParam),Hparam_shifted.Mesh,param.MG,TYPE,nrhs,param.verbose);
		
		## THIS CODE USES GEOMETRIC MULTIGRID - RAP style.
		MGsetup(ShiftedHT,param.helmParam.Mesh,param.MG,TYPE,nrhs,param.verbose);
	end

	if (doTranspose != param.MG.doTranspose)
		transposeHierarchy(param.MG);
	end
	
	adjustMemoryForNumRHS(param.MG,TYPE,size(B,2),param.verbose)
	BLAS.set_num_threads(param.MG.numCores);
	ShiftedHT = param.MG.As[1];
	Az = param.MG.memCycle[1].b;
	if param.shift[1] != 0.0
		MShift = GetHelmholtzShiftOP(param.helmParam.m, param.helmParam.omega,param.shift[1]);
		if doTranspose==1
			MShift = -MShift; # this is because of the conjugate			
		end
		Afun = getHelmholtzFun(ShiftedHT,MShift,Az,param.MG.numCores);
	else
		Afun = getAfun(ShiftedHT,Az,param.MG.numCores);
	end
	
	param.setupTime += (time_ns() - tt)/(10^9);
	tt = time_ns()
	if param.Krylov=="GMRES"
		X, param.MG,num_iter = solveGMRES_MG(Afun,param.MG,B,Array{eltype(B)}(undef,0),param.verbose,param.inner);
		param.nPrec += num_iter;
	elseif param.Krylov=="BiCGSTAB"
		X, param.MG,num_iter,nprec = solveBiCGSTAB_MG(Afun,param.MG,B,Array{eltype(B)}(undef,0),param.verbose);
		param.nPrec += nprec;
	end
	param.solveTime +=(time_ns() - tt)/(10^9);
	
	if num_iter >= param.MG.maxOuterIter
		warn("MG solver reached maximum iterations without convergence");
	end

	return X, param
end

import jInv.Utils.clear!
function clear!(s::ShiftedLaplacianMultigridSolver)
	 clear!(s.MG);
	 clear!(s.helmParam)
	 s.doClear = 0;
end