using jInv.Mesh;
using ForwardHelmholtz
using ForwardHelmholtz.PointSourceADR
using Multigrid
using MAT
using EikonalInv

plotting = false;
if plotting
	using jInvVis
	using PyPlot;
	close("all");
end
include("getModels.jl");
function performExperiment(formulation::Int64,n,f,model,coarseSolveType = "MUMPS")
n_tup = tuple(n...);
dim = length(n);

pad    = 25;
if model=="linear"
	domain = [0.0,20.0,0.0,4.0];
	if dim == 3
		domain = [0.0,20.0,0.0,20.0,0.0,4.0];
	end
	m = getLinearModel(1.6,3.5,n);
	m = 1./(m.^2);
elseif model=="Marmousi"
	if dim==3
		error("Marmousi is only 2D");
	end
	domain = [0.0,17.0,0.0,3.5];
	model_dir = "./../../../BenchmarkModels/";
	file = matopen(string(model_dir,"ElasticMarmousi2_single.mat")); 
	DICT = read(file); close(file);
	m = DICT["Vp"]';
	
	m = expandModelNearest(m,[13601;2801],n);
	m = 1./m.^2;
	# ms = copy(m);
	# for k=1:10
		# for j = 2:size(m,2)-1
			# for i = 2:size(m,1)-1
				# @inbounds ms[i,j] = (2*ms[i,j] + (ms[i-1,j-1]+ms[i-1,j]+ms[i-1,j+1]+ms[i,j-1]+ms[i,j+1]+ms[i+1,j-1]+ms[i+1,j]+ms[i,j+1]))/10.0;
			# end
		# end
	# end
	# m = ms;
elseif model=="Overthrust"
	domain = [0.0,20.0,0.0,20.0,0.0,4.65];
	model_dir = "./../../../BenchmarkModels/";
	file = matopen(string(model_dir,"3DOverthrust801801187.mat")); 
	DICT = read(file); close(file);
	m = DICT["A"];DICT = 0;
	m = m.*1e-3;
	m = reshape(m,801,801,187);
	m = expandModelNearest(m,[801;801;187],n);
	m = 1./m.^2;

elseif model=="Wedge"
	if dim==3
		error("TBD");
	end
	n_cells = n-1; (m,Minv) = getWedgeModel(n);domain = Minv.domain;
else
	domain = [0.0,1.0,0.0,1.0];
	if dim == 3
		domain = [0.0,1.0,0.0,1.0,0.0,1.0];
	end
	m = ones(n_tup);
end

Minv = getRegularMesh(domain,n-1);
println("RUN: formulation = ",formulation," n = ",n," f = ",f," model = ",model," coarsest = ",coarseSolveType);

## Generating the right hand side
if dim==3
	q = zeros(Complex64,n_tup);
	src = [div(Minv.n[1]+1,2),div(Minv.n[2]+1,2),1];
	q[src[1],src[2],src[3]] = 1./(prod(Minv.h))
else
	q = zeros(Complex64,n_tup);
	src = [div(Minv.n[1]+1,2),1];
	q[src[1],src[2]] = 1./(prod(Minv.h))
end

q = q[:];

w = 2*pi*f

# if model!="const"
	# w = getMaximalFrequency(m,Minv)*0.95;
# end
println("maximal frequency is: f = ",getMaximalFrequency(m,Minv)/(2*pi));
println("sp*omega*h = ",w*maximum(Minv.h)*maximum(sqrt(m)));

if plotting
	figure()
	plotModel(1./sqrt(m),true,Minv,0);
	# return;
end

pad = pad*ones(Int64,3);
maxOmega = getMaximalFrequency(m,Minv);
ABLamp = maxOmega;
NeumannAtFirstDim = true;
gamma = 0.000001*2*pi*ones(Float32,size(m));

# (ADR_long,ADR_short,T,G1,G2) = getHelmholtzADR(Minv, m, w, gamma,NeumannAtFirstDim,pad,ABLamp,src,true);

# H,gamma = GetHelmholtzOperator(Minv, m, w, gamma,NeumannAtFirstDim,pad,ABLamp,true);

shift 		= 0.2;
Shift 		= GetHelmholtzShiftOP(m, w,shift)
levels      = 5;
numCores 	= 4; 
maxIter     = 50;
relativeTol = 1e-5;
cycleType   ='K';
relaxType   = "Jac-GMRES";
relaxParam  = 0.8;
relaxPre 	= l->l+1;
relaxPost   = l->l+1;

if formulation == 1 # Pure Helmholtz
	# relaxPre 	= 2;
	# relaxPost   = 2;
	Shift = Shift';
	if eltype(q)==Complex64
		Shift = convert(SparseMatrixCSC{Complex64,Int64},Shift);
	end
		
	gamma += getABL(Minv,NeumannAtFirstDim,pad,ABLamp);
	######################################
	######################################
	# tic()
	# H_param = HelmholtzParam(Minv,gamma,m,w,NeumannAtFirstDim,true);
	# Hparam_shifted = getShiftedHelmholtzParam(H_param,shift);
	# function restrictParam(mesh_fine,mesh_coarse,param_fine,level)
		# P = getShiftedHelmholtzParam(param_fine,0.0); 
		# P.gamma = restrictNodalVariables(P.gamma,mesh_fine.n+1);
		# P.m = restrictNodalVariables(P.m,mesh_fine.n+1); 
		# P.Mesh = mesh_coarse; 
		# return P;
	# end
	# getOperator = (mesh,Hparam)->GetHelmholtzOperator(mesh,Hparam.m,Hparam.omega,Hparam.gamma,Hparam.NeumannOnTop,Hparam.Sommerfeld);
	# MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				# relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,"FullWeighting");
	# MGsetup(getMultilevelOperatorConstructor(Hparam_shifted,getOperator,restrictParam),Hparam_shifted.Mesh,MG,eltype(q),1,true);
	# SHT = MG.As[1];
	# set1 = toc();
	# set2 = 0.0;
	#########################################
	#########################################
	println("getting Helmholtz operator")
	tic()
	HT = GetHelmholtzOperator(Minv, m, w, gamma,NeumannAtFirstDim,pad,ABLamp,true)[1]';
	if eltype(q)==Complex64
		HT = convert(SparseMatrixCSC{Complex64,Int64},HT);
	end
	
	SHT = HT + Shift;HT = 0.0;
	set1 = toc()
	println("setup")
	tic()
	MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,"FullWeighting");
	MGsetup(SHT,Minv,MG,eltype(q),1,false);
	set2 = toc();
	#########################################
	#########################################

	HT = x->SHT'*x - Shift'*x;
	println("FGMRES-MG")
	tic()
	x = solveGMRES_MG(HT,MG,q,zeros(eltype(q),size(q)),true,5)[1]
	sol1 = toc()
	println(norm(HT(x) - q)/norm(q))
	
	println("Total setup = ", set1 + set2 );
	println("Total sol = ", sol1 );
elseif formulation == 4 # ADR long up
	println("problem disc.")
	gamma += getABL(Minv,NeumannAtFirstDim,pad,ABLamp);
	#############################################################
	###########################################################
	# tic()
	# T,G,LT = getADRcoefficients(Minv, m,src);
	# set1 = toc();
	# tic()
	# (ADR_longUp,~,~) = getHelmholtzADR(false,Minv, m, w, gamma,T,G,LT,NeumannAtFirstDim,src,true);
	# ADR_longUp = ADR_longUp';
	# ADRparam = getADRparam(Minv,gamma,m,w,T,G,LT,src,NeumannAtFirstDim,true);
	# getOperator = (mesh,ADRparam)->((ADR2,ADR1,~) = getHelmholtzADR(false,ADRparam.Minv, ADRparam.m, ADRparam.w, ADRparam.gamma,
									# ADRparam.T,ADRparam.G,ADRparam.LT,ADRparam.NeumannAtFirstDim,ADRparam.src,ADRparam.Sommerfeld);
									# return 0.25*ADR1 + 0.75*ADR2;);
	# MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				# relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,"FullWeighting");
	
	# MGsetup(getMultilevelOperatorConstructor(ADRparam,getOperator,restrictADRParam),ADRparam.Minv,MG,eltype(q),1,true);
	# SHT = MG.As[1];
	# set2 = toc();
	
	###########################################################
	############################################################
	tic()
	T,G,LT = getADRcoefficients(Minv, m,src);
	set1 = toc();
	
	# tic()
	(ADR_longUp,ADR_short,~) = getHelmholtzADR(false,Minv, m, w, gamma,T,G,LT,NeumannAtFirstDim,src,true);
	T = 0; G = 0; LT = 0; m = 0; gamma = 0;
	# set1 = toc()
	println("setup")
	tic()
	ADR_longUp = ADR_longUp';
	if eltype(q)==Complex64
		ADR_short = convert(SparseMatrixCSC{Complex64,Int64},0.25*ADR_short') + convert(SparseMatrixCSC{Complex64,Int64},0.75*ADR_longUp);
	else
		ADR_short = 0.25*ADR_short' + 0.75*ADR_longUp;
	end
	
	
	MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,"FullWeighting");
	MGsetup(ADR_short,Minv,MG,eltype(q),1,false);
	set2 = toc()
	######################################################################
	#######################################################################3
	println("FGMRES-MG")
	tic()
	x = solveGMRES_MG(ADR_longUp,MG,q,zeros(eltype(q),size(q)),true,5)[1]
	sol1 = toc()
	println(norm(ADR_longUp'*x - q)/norm(q))
	println("Total setup = ", set1 + set2 );
	println("Total sol = ", sol1 );
elseif formulation == 5 # ADR long shifted as prec to ADR long
	# (ADR_long,ADR_short,T) = getHelmholtzADR(true,Minv, m, w, gamma,NeumannAtFirstDim,pad,ABLamp,src,true);
	# Shift = Shift'
	# M = exp(-1im*w*T);
	# ADR_long = (spdiagm(M)*ADR_long*spdiagm(1./M));
	# SHT = ADR_long' + Shift;ADR_long = 0.0;
	# MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				# relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,"FullWeighting");
	# MGsetup(SHT,Minv,MG,eltype(q),1,true);
	# HT = x->SHT'*x - Shift'*x;
	# println("FGMRES-MG")
	# x = solveGMRES_MG(HT,MG,q,zeros(eltype(q),size(q)),true,5)[1]
	# println(norm(HT(x) - q)/norm(q))
elseif formulation == 8 # two-phase solution.
	# println("problem disc.")
	gamma += getABL(Minv,NeumannAtFirstDim,pad,ABLamp);
	# #############################################################
	# ###########################################################
	tic()
	T,G,LT = getADRcoefficients(Minv, m,src);
	set1 = toc();
	tic()
	ADRparam = getADRparam(Minv,gamma,m,w,T,G,LT,src,NeumannAtFirstDim,true);
	getOperator = (mesh,ADRparam)->((ADR2,ADR1,~) = getHelmholtzADR(true,ADRparam.Minv, ADRparam.m, ADRparam.w, ADRparam.gamma,
									ADRparam.T,ADRparam.G,ADRparam.LT,ADRparam.NeumannAtFirstDim,ADRparam.src,ADRparam.Sommerfeld);
									return ADR1);
	MG = getMGparam(levels,numCores,1,0.01,relaxType,relaxParam,
				relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,"FullWeighting");
	MGsetup(getMultilevelOperatorConstructor(ADRparam,getOperator,restrictADRParam),ADRparam.Minv,MG,eltype(q),1,true);
	ADRparam = 0;
	ADR_short = MG.As[1];
	set2 = toc();
	# #############################################################
	# ###########################################################
	# println("problem disc.")
	# tic()
	# (ADR_long,ADR_short,T) = getHelmholtzADR(true,Minv, m, w, gamma,NeumannAtFirstDim,pad,ABLamp,src,true);
	# set1 = toc()
	
	# # Lap = ForwardHelmholtz.getNodalLaplacianMatrix(Minv);
	# # HT = GetHelmholtzOperator(Minv, m, w, gamma,NeumannAtFirstDim,pad,ABLamp,true)[1]';
	# # HT = (spdiagm(1./M)*(HT)*spdiagm(M));
	# println("Setup:")
	# tic()
	# M = exp(-1im*w*T);
	# # ADR_short = (0.25*ADR_short+0.75*ADR_long)';
	# ADR_short = ADR_short';
	# ADR_long = ADR_long';
	# toc()
	# tic()
	# MG = getMGparam(levels,numCores,1,0.01,relaxType,relaxParam,
		# relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,"FullWeighting");
	# MGsetup(ADR_short,Minv,MG,eltype(q),1,false);
	# set2 = toc()
	# #############################################################
	# ###########################################################
	
	println("FGMRES-MG")
	tic()
	x = solveGMRES_MG(ADR_short,MG,q,zeros(eltype(q),size(q)),true,5)[1]
	sol1 = toc()
	# println(norm(ADR_long'*x - q)/norm(q))
	
	ADR_short = 0;
	MG.As[1] = MG.As[end];
	MG.As[2] = MG.As[end];
	MG.As[3] = MG.As[end];
	# x = zeros(eltype(q),size(q));
	
	# #############################################################
	# ###########################################################
	
	# HT = GetHelmholtzOperator(Minv, m, w, gamma,NeumannAtFirstDim,pad,ABLamp,true)[1]';
	# HT = (spdiagm(M)*(HT)*spdiagm(1./M));
	# Shift = Shift';
	# println("Setup II :")
	# tic()
	# rescaleMatrix(ADR_long,M); # HT = (spdiagm(M)*(ADR_long)*spdiagm(1./M));
	# HT = ADR_long;
	# x = x.*M;
	# relres = norm(HT'*x - q)/norm(q);
	# println("Rescaling the problem, res:", relres);
	# MG.relativeTol = (relativeTol/relres);
	# MG.maxOuterIter = maxIter;
	# SHT = HT + Shift;HT = 0.0;
	# replaceMatrixInHierarchy(MG,SHT);
	# HT = x->SHT'*x - Shift'*x;
	# set3 = toc()
	# #############################################################
	# ###########################################################
	
	
	
	
	(ADR_long,~,~) = getHelmholtzADR(true,Minv, m, w, gamma,T,G,LT,NeumannAtFirstDim,src,true);
	G = 0; LT = 0; m = 0; gamma = 0;
	ADR_long = ADR_long';
	tic()
	relres = norm(ADR_long'*x - q)/norm(q);
	println("Rescaling the problem, res:", relres);
	
	M = exp(-1im*w*T);
	if eltype(q)==Complex64
		M = convert(Array{Complex64},M);
	end
	T = 0;
	
	
	x = x.*M;
	rescaleMatrix(ADR_long,M);
	Shift = Shift';
	if eltype(q)==Complex64
		Shift = convert(SparseMatrixCSC{Complex64,Int64},Shift);
	end
	###############################################
	###############################################3
	# ADRparam = getADRparam(Minv,gamma,m,w,T,G,LT,src,NeumannAtFirstDim,true,M);
	# getOperator = (mesh,ADRparam)->((ADR2,ADR1,~) = getHelmholtzADR(true,ADRparam.Minv, ADRparam.m, ADRparam.w, ADRparam.gamma,
									# ADRparam.T,ADRparam.G,ADRparam.LT,ADRparam.NeumannAtFirstDim,ADRparam.src,ADRparam.Sommerfeld);
									# H = ADR2 + GetHelmholtzShiftOP(ADRparam.m, ADRparam.w,shift);
									# # M = exp(-1im*w*T);
									# # M = ADRparam.scaling;
									# # rescaleMatrix(H,M);
									# return H);
									
	
	# MG = getMGparam(levels,numCores,maxIter,relativeTol/relres,relaxType,relaxParam,
				# relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,"FullWeighting");
	# MGsetup(getMultilevelOperatorConstructor(ADRparam,getOperator,restrictADRParam),ADRparam.Minv,MG,eltype(q),1,true);
	# ADR_long = MG.As[1];
	# HT = x->MG.As[1]'*x - Shift'*x;
	###############################################
	###############################################
	MG.relativeTol = (relativeTol/relres);
	MG.maxOuterIter = maxIter;
	ADR_long_shifted = ADR_long + Shift;ADR_long = 0;
	replaceMatrixInHierarchy(MG,ADR_long_shifted);
	HT = x->MG.As[1]'*x - Shift'*x;
	
	set3 = toc();
	
	println("FGMRES-MG")
	tic()
	x = solveGMRES_MG(HT,MG,q,x,true,5)[1]
	sol2 = toc();
	
	println("Total setup = ", set1 + set2 + set3);
	println("Total sol = ", sol1 + sol2);
	println(norm(HT(x) - q)/norm(q))
end
println("**************************************** THE END ***************************************************");
end



if true
println("*********************************************************************")
println("********************** WARM-UP  ************************************")
println("*********************************************************************")
n = [65;65;33];
performExperiment(1,n,0.5,"linear","GMRES");
performExperiment(8,n,0.5,"linear","GMRES");
performExperiment(4,n,0.5,"linear","GMRES");

performExperiment(1,n,0.5,"Overthrust","GMRES");
performExperiment(8,n,0.5,"Overthrust","GMRES");
performExperiment(4,n,0.5,"Overthrust","GMRES");


# n = [257;257;65];
# performExperiment(1,n,1.5,"linear","GMRES");
# performExperiment(8,n,1.5,"linear","GMRES");
# performExperiment(4,n,1.5,"linear","GMRES");
# println("*********************************************************************")

# performExperiment(1,n,2.0,"linear","GMRES");
# performExperiment(8,n,2.0,"linear","GMRES");
# performExperiment(4,n,2.0,"linear","GMRES");



# n = [385;385;97];
# performExperiment(1,n,2.0,"linear","GMRES");
# performExperiment(8,n,2.0,"linear","GMRES");
# performExperiment(4,n,2.0,"linear","GMRES");
# println("*********************************************************************")

# performExperiment(1,n,3.0,"linear","GMRES");
# performExperiment(8,n,3.0,"linear","GMRES");
# performExperiment(4,n,3.0,"linear","GMRES");
# println("*********************************************************************")

 n = [513;513;129];
 performExperiment(1,n,3.0,"linear","GMRES");
 performExperiment(8,n,3.0,"linear","GMRES");
 performExperiment(4,n,3.0,"linear","GMRES");
 println("*********************************************************************")

 performExperiment(1,n,4.0,"linear","GMRES");
 performExperiment(8,n,4.0,"linear","GMRES");
 performExperiment(4,n,4.0,"linear","GMRES");
 println("*********************************************************************")

# n = [769;769;193];
# performExperiment(1,n,4.0,"linear","GMRES");
# performExperiment(8,n,4.0,"linear","GMRES");
# performExperiment(4,n,4.0,"linear","GMRES");
# println("*********************************************************************")

# performExperiment(1,n,6.0,"linear","GMRES");
# performExperiment(8,n,6.0,"linear","GMRES");
# performExperiment(4,n,6.0,"linear","GMRES");
# println("*********************************************************************")


n = [257;257;65];
performExperiment(1,n,2.0,"Overthrust","GMRES");
performExperiment(8,n,2.0,"Overthrust","GMRES");
performExperiment(4,n,2.0,"Overthrust","GMRES");
println("*********************************************************************")

performExperiment(1,n,3.0,"Overthrust","GMRES");
performExperiment(8,n,3.0,"Overthrust","GMRES");
performExperiment(4,n,3.0,"Overthrust","GMRES");



n = [385;385;97];
performExperiment(1,n,3.0,"Overthrust","GMRES");
performExperiment(8,n,3.0,"Overthrust","GMRES");
performExperiment(4,n,3.0,"Overthrust","GMRES");
println("*********************************************************************")

performExperiment(1,n,4.0,"Overthrust","GMRES");
performExperiment(8,n,4.0,"Overthrust","GMRES");
performExperiment(4,n,4.0,"Overthrust","GMRES");
println("*********************************************************************")


n = [513;513;129];
performExperiment(1,n,4.0,"Overthrust","GMRES");
performExperiment(8,n,4.0,"Overthrust","GMRES");
performExperiment(4,n,4.0,"Overthrust","GMRES");
println("*********************************************************************")

performExperiment(1,n,6.0,"Overthrust","GMRES");
performExperiment(8,n,6.0,"Overthrust","GMRES");
performExperiment(4,n,6.0,"Overthrust","GMRES");
println("*********************************************************************")

end
# 2D results:

if false

println("*********************************************************************")
println("********************** WARM-UP  ************************************")
println("*********************************************************************")

n = [257;129];
performExperiment(1,n,2.0,"linear","GMRES");
performExperiment(8,n,2.0,"linear","GMRES");
performExperiment(4,n,2.0,"linear","GMRES");





println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")



n = [769;257];
performExperiment(1,n,3.5,"linear","GMRES");
performExperiment(8,n,3.5,"linear","GMRES");
performExperiment(4,n,3.5,"linear","GMRES");
println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")

performExperiment(1,n,5.5,"linear","GMRES");
performExperiment(8,n,5.5,"linear","GMRES");
performExperiment(4,n,5.5,"linear","GMRES");

println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")

n = [1025;385];
performExperiment(1,n,5.5,"linear","GMRES");
performExperiment(8,n,5.5,"linear","GMRES");
performExperiment(4,n,5.5,"linear","GMRES");

println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")

performExperiment(1,n,7.5,"linear","GMRES");
performExperiment(8,n,7.5,"linear","GMRES");
performExperiment(4,n,7.5,"linear","GMRES");

# println("*********************************************************************")
# println("*********************************************************************")
# println("*********************************************************************")

n = [1537;513];
performExperiment(1,n,7.5,"linear","GMRES");
performExperiment(8,n,7.5,"linear","GMRES");
performExperiment(4,n,7.5,"linear","GMRES");


performExperiment(1,n,11.0,"linear","GMRES");
performExperiment(8,n,11.0,"linear","GMRES");
# performExperiment(9,n,11.0,"linear","GMRES");
performExperiment(4,n,11.0,"linear","GMRES");



println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")

n = [769;257];
performExperiment(1,n,3.0,"Marmousi","GMRES");
performExperiment(8,n,3.0,"Marmousi","GMRES");
performExperiment(4,n,3.0,"Marmousi","GMRES");

println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")

performExperiment(1,n,4.5,"Marmousi","GMRES");
performExperiment(8,n,4.5,"Marmousi","GMRES");
performExperiment(4,n,4.5,"Marmousi","GMRES");

println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")

n = [1025;385];
performExperiment(1,n,4.5,"Marmousi","GMRES");
performExperiment(8,n,4.5,"Marmousi","GMRES");
performExperiment(4,n,4.5,"Marmousi","GMRES");

println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")


performExperiment(1,n,6.5,"Marmousi","GMRES");
performExperiment(8,n,6.5,"Marmousi","GMRES");
performExperiment(4,n,6.5,"Marmousi","GMRES");

println("*********************************************************************")
println("*********************************************************************")
println("*********************************************************************")


n = [1537;513];
performExperiment(1,n,6.5,"Marmousi","GMRES");
performExperiment(8,n,6.5,"Marmousi","GMRES");
performExperiment(4,n,6.5,"Marmousi","GMRES");

# println("*********************************************************************")
# println("*********************************************************************")
# println("*********************************************************************")

performExperiment(1,n,9.0,"Marmousi","GMRES");
performExperiment(8,n,9.0,"Marmousi","GMRES");
performExperiment(4,n,9.0,"Marmousi","GMRES");
end


