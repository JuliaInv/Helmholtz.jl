using jInv.Mesh;
using Helmholtz
using Multigrid
using LinearAlgebra
using SparseArrays

const plotting = false;

if plotting
	using PyPlot;
	close("all");
end

# m = readdlm("SEGmodel2Dsalt.dat"); m = Matrix(m'); m = m*1e-3;
m = 1.5*ones(257,129);


Minv = getRegularMesh([0.0,13.5,0.0,4.2],collect(size(m)).-1);

pad = 16;


m = 1.0./m.^2

f = 2.5;

w = 2*pi*f

println("omega*h:");
println(w*Minv.h*sqrt(maximum(m)));
pad = pad*ones(Int64,Minv.dim);

maxOmega = getMaximalFrequency(m,Minv);
ABLamp = maxOmega;


Sommerfeld = true;
NeumannAtFirstDim = true;
H,gamma = GetHelmholtzOperator(Minv,m,w,w*ones(size(m))*0.01,NeumannAtFirstDim,pad,ABLamp,Sommerfeld);
shift = [0.2;0.2;0.2]./10.0;
SH = H .+ GetHelmholtzShiftOP(m, real(w),shift[1]);



# n = Minv.n .+1 ; n_tup = tuple(n...);
# src = div.(n,2);
# src[end] = 1;
# q = zeros(ComplexF64,n_tup)
# q[loc2cs(n,src)] = 1.0/(Minv.h[1]^2);

levels      = 2;
numCores 	= 2; 
maxIter     = 30;
relativeTol = 1e-6;
# relaxType   = "Jac-GMRES";
relaxType   = "Jac";
relaxParam  = 0.75;
relaxPre 	= 2;
relaxPost   = 2;
cycleType   ='W';
coarseSolveType = "NoMUMPS";

MG = getMGparam(ComplexF64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0)


				
## Preparing a point shource RHS ############################
n = Minv.n .+ 1; n_tup = tuple(n...);
src = div.(n,2);
src[end] = 1;
q = zeros(ComplexF64,n_tup)
q[loc2cs(n,src)] = 1.0/(Minv.h[1]^2);
###############################################
b = q[:];

Hparam = HelmholtzParam(Minv,gamma,vec(m),w,NeumannAtFirstDim,Sommerfeld);
Ainv = getShiftedLaplacianMultigridSolver(Hparam, MG,shift,"GMRES",20,true);


Ainv = copySolver(Ainv);

x = solveLinearSystem(sparse(SH'),b,Ainv)[1];
println(norm(H*x - b)/norm(b))

MG.relaxType = "Jac";
MG.cycleType = 'W';
Ainv = getShiftedLaplacianMultigridSolver(Hparam, MG,shift,"BiCGSTAB",0,true);
x = solveLinearSystem(sparse(SH'),b,Ainv)[1];

if plotting
	reX = real(reshape(x,n_tup));
	reX[My_sub2ind(n,src)] = 0.0;
	println(norm(q[:]-H*x)/norm(q[:]));
	println(norm(q[:]-H*y)/norm(q[:]));
	figure();
	imshow(reX');title("Helmholtz Iterative Solution");
end

if plotting
	s = H\q[:];
	s = real(reshape(s,n_tup));
	s[My_sub2ind(n,src)] = 0.0;
	figure();
	imshow(s');title("Helmholtz True Solution");
end

# println("Doing Transpose")
# y = solveLinearSystem(sparse(SH'),b,Ainv,1)[1];

# println(norm(H'*y-b)/norm(b));


################## WITHOUT THE COARSEST GRID SOL ########################
# coarseSolveType = "DDNodal";
# MG = getMGparam(ComplexF64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				# relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0)
# Ainv = getShiftedLaplacianMultigridSolver(Hparam, MG,shift,"BiCGSTAB",0,true);


# y = solveLinearSystem(SH',b,Ainv)[1];
###########################################################################


println("SOLVING MULTIPLE RHSs")
coarseSolveType = "Julia";
MG = getMGparam(ComplexF64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0)
Ainv = getShiftedLaplacianMultigridSolver(Hparam, MG,shift,"BiCGSTAB",0,true);
nrhs = 2;
b = rand(ComplexF64,length(b),nrhs);
clear!(Ainv);
Ainv.helmParam = Hparam;
x = solveLinearSystem(sparse(SH'),b,Ainv)[1];
println(norm(H*x .- b)/norm(b))

MG = getMGparam(ComplexF64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType);
MG.relaxType = "Jac-GMRES";
MG.cycleType = 'K';
Ainv = getShiftedLaplacianMultigridSolver(Hparam, MG,shift,"GMRES",5,true);
x = solveLinearSystem(sparse(SH'),b,Ainv)[1];
println(norm(H*x .- b)/norm(b))
