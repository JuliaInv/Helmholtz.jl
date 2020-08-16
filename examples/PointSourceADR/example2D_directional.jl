using jInv.Mesh;
using Helmholtz
using Helmholtz.PointSourceADR
using Multigrid

plotting = true;

if plotting
	using PyPlot;
	close("all");
end

include("getModels.jl");

NeumannAtFirstDim = false;
Sommerfeld = true;
gamma_val = 0.01;

model = "const";
domain = [0.0,1.0,0.0,1.0];
n_cells = [512,512];
src = div.(n_cells,2).+1;
# src[2] = 1;

if model=="const"
	NeumannAtFirstDim = false;
	Sommerfeld = true;
	f = 40.0;
	src = div.(n_cells,2).+1;
	m_coarse = getConstModel(n_cells.+1);
	pad_cells = [20;20];
else
	error("unknown model");
end
Minv = getRegularMesh(domain,n_cells);



w = 2*pi*f

if plotting
	figure()
	imshow(m_coarse'); colorbar();
	title(string("The ",model," model"))
end


n_tup = tuple((n_cells.+1)...);
q_coarse = zeros(ComplexF64,n_tup)
src2 = div.(n_cells,2).+1;
q_coarse[src2[1],src2[2]] = 1/(Minv.h[2]^2);

# src2 = div(3*n_cells,4)+1;
# q_coarse[src2[1],src2[2]] = 1/(Minv.h[2]^2);
# src = src2;


println("omega*h:");
println(w*Minv.h*sqrt(maximum(m_coarse)));

maxOmega = getMaximalFrequency(m_coarse,Minv);
ABLamp = maxOmega;


m = m_coarse;


gamma = (w*gamma_val)*ones(Float32,size(m)) + getABL(Minv.n.+1,NeumannAtFirstDim,pad_cells.+1,ABLamp);


if plotting
	figure();
	imshow(reshape(gamma,n_tup));
end


# getHelmholtzADR(central::Bool, Mesh::RegularMesh, mNodal::Array{Float64},omega::Float64, gamma::Array,T,G,LT,
									# NeumannAtFirstDim::Bool,src::Array{Int64},Sommerfeld::Bool,single::Bool = true)

X1,X2 = ndgrid(collect(0:Minv.n[1])*Minv.h[1],collect(0:Minv.n[2])*Minv.h[2]);

T  = X1;
Tx = ones(size(X1));
Ty = zeros(size(X1));
# G = [Tx,Ty];
G =  Array{Array{Float64}}(undef, Minv.dim);
G[1] = Tx;
G[2] = Ty;
LT = zeros(size(X1));
(ADR_long,ADR_short,T) = getHelmholtzADR(true,Minv, m, w, gamma,T,G,LT, NeumannAtFirstDim,[3,3],Sommerfeld);

# ADR_short = 0.5*ADR_long + 0.5*ADR_short;

if plotting
	figure();
	imshow(real(exp.(1im*w*reshape(T,n_tup)))');
end

H, = GetHelmholtzOperator(Minv, m, w, gamma_val*ones(Float32,size(m)),NeumannAtFirstDim,pad_cells.+1,ABLamp,Sommerfeld);
S =  GetHelmholtzShiftOP(m, w,0.2);

q = q_coarse;
xh = (H\q[:])
q = q[:].*exp.(1im*w*T[:]);
xs = ((ADR_short )\q[:]).*exp.(-1im*w*T[:]);
us = (H+S)\q[:];



# if plotting
# figure();
# subplot(1,4,1);
# imshow(real(I2'));
# subplot(1,4,2);
# imshow(reshape(T,n_tup)');
# subplot(1,4,3);
# imshow(reshape(real(alu),n_tup)');
# subplot(1,4,4);
# imshow(reshape(real(exp.(-1im*w*T)),n_tup)');

if plotting
	figure();
	subplot(1,3,1);
	imshow(reshape(real(xh),n_tup)',cmap = "jet"); title("Full Helmholtz solution"); colorbar();
	subplot(1,3,2);
	imshow(reshape(real(-us),n_tup)',cmap = "jet"); title("Shifted Helmholtz solution"); colorbar();
	subplot(1,3,3);
	imshow(reshape(real(xs),n_tup)',cmap = "jet"); title("First order upwind ADR solution");colorbar();
end
a = 1;


levels      = 6;
numCores 	= 2; 
maxIter     = 10;
relativeTol = 1e-10;
relaxType   = "Jac-GMRES";
relaxParam  = 0.7;
relaxPre 	= 3;
relaxPost   = 3;
cycleType   ='W';
coarseSolveType = "NoMUMPS";

MG = getMGparam(ComplexF64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0);
nrhs = 1;
MGsetup(sparse(ADR_short'),Minv,MG,nrhs,true);

println("****************************** Stand-alone GMG RAP for ADR 1st: ******************************")
a = zeros(ComplexF64, size(q[:]));
solveMG(MG,q,a,true);

println("****************************** GMRES GMG RAP for ADR 1st: ******************************")
F = getAfun(sparse(ADR_short'),zeros(eltype(q),size(q)),1)
MG.maxOuterIter = 2;
solveGMRES_MG(F,MG,q,a,true,10,true);



z = 1;


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

