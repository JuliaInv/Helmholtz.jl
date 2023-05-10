using jInv.Mesh
using Multigrid
using Multigrid.DomainDecomposition
using SparseArrays
using LinearAlgebra






domain = [0.0, 1.0, 0.0, 1.0];
n = [64,64];
Mr = getRegularMesh(domain,n)

pad = 8;
m = ones(tuple(n.+1...));

maxOmega = getMaximalFrequency(m,Minv);
ABLamp = maxOmega;

w = maxOmega;

println("omega*h:");
println(w*Minv.h*sqrt(maximum(m)));
pad = pad*ones(Int64,Minv.dim);


println("************************************************* Hybrid Keczmarz for Acoustic Helmholtz 2D ******************************************************");

Sommerfeld = true;
NeumannAtFirstDim = true;
H,gamma = GetHelmholtzOperator(Minv,m,w,w*ones(size(m))*0.01,NeumannAtFirstDim,pad,ABLamp,Sommerfeld);
shift = 0.4;
SHT = sparse((H .+ GetHelmholtzShiftOP(m, real(w),shift))');


numCores = 2;
innerKatzIter = 5;
iter = 5;
outer = 2;


q = getAcousticPointSource(Mr,ComplexF64);
x = zeros(ComplexF64,size(q));
N = size(Ar,2);

HKparam = getHybridKaczmarz(ComplexF64,Int64, SHT,Mr,[4,4], getNodalIndicesOfCell,0.9,numCores,innerKatzIter);
prec = getHybridKaczmarzPrecond(HKparam,SH);
x = FGMRES_relaxation(SHT,copy(q),x,inner,prec,1e-3,true,true,numCores)[1]
println("Norm is: ",norm(SHT'*x - q))

# println("************************************************* Hybrid Keczmarz for Elastic Helmholtz 2D ******************************************************");

# mu = 2.0*ones(prod(Mr.n));
# lambda = mu;
# Ar = GetLinearElasticityOperator(Mr,mu,lambda);
# Ar = Ar + 2e-1*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
# N = size(Ar,2);
# b = Ar*rand(N); b = b/norm(b);
# x = zeros(N);

# HKparam = getHybridKaczmarz(Float64,Int64, Ar,Mr,[4,4], getFacesStaggeredIndicesOfCellNoPressure,0.8,numCores,innerKatzIter);
# prec = getHybridKaczmarzPrecond(HKparam,Ar);
# x = FGMRES_relaxation(Ar,copy(b),x,inner,prec,1e-5*norm(b),true,true,numCores)[1]
# println("Norm is: ",norm(Ar*x - b))











# using KrylovMethods
# domain = [0.0, 1.0, 0.0, 1.0];
# n = [128,128];
# Mr = getRegularMesh(domain,n)
# numCores = 8;
# innerKatzIter = 200;
# omega = 4.0*getMaximalFrequency(ones(10,1),Mr);
# spValType = Complex64;
# gamma = 0.25;
# inner = 5;



# m = ones(tuple(Mr.n+1...));
# # AcT = convert(SparseMatrixCSC{spValType,Int64},GetHelmholtzOperator(Mr, m,omega , 0.0001*m,false,[10,10],omega*1.0,false)[1])';
# McT = convert(SparseMatrixCSC{spValType,Int64},GetHelmholtzOperator(Mr, m,omega , omega*gamma*m,false,[10,10],omega*1.0,false)[1])';
# q = rand(spValType,size(McT,2));
# x = zeros(spValType,size(McT,2));

# HKparam = getHybridKaczmarz(McT,Mr,[8,8], getNodalIndicesOfCell,1.1,numCores,innerKatzIter);
# prec = getHybridKaczmarzPrecond(HKparam,McT,q);

# y = zeros(spValType,size(McT,2));
# # y = FGMRES(McT,copy(q),y,inner,prec,1e-5*norm(q),true,true,numCores)[1]
# println("Norm is: ",norm(McT'*y - q))



# println("Elastic Experiment")
# gamma = 0.25;
# m = ones(tuple(Mr.n...));
# EHparam = ElasticHelmholtzParam(Mr,omega,2*m,m,m,gamma*m,false,false);
# McT = convert(SparseMatrixCSC{spValType,Int64},GetElasticHelmholtzOperator(EHparam))';
# q = rand(spValType,size(McT,2));
# x = zeros(spValType,size(McT,2));
# HKparam = getHybridKaczmarz(McT,Mr,[8,8], getFacesStaggeredIndicesOfCellNoPressure,0.85,numCores,innerKatzIter);
# prec = getHybridKaczmarzPrecond(HKparam,McT,q);
# y = zeros(spValType,size(McT,2));
# z = prec(q);
# println("Norm is: ",norm(McT'*z - q)/norm(q))

# y = FGMRES(McT,copy(q),y,inner,prec,1e-5*norm(q),true,true,numCores)[1]
# println("Norm is: ",norm(McT'*y - q))



# innerKatzIter = 4;
# println("Elastic Experiment Mixed")
# gamma = 0.25;
# m = ones(tuple(Mr.n...));
# mixedFormulation = true;
# Kaczmarz = false;
# println("Kaczmarz = ",Kaczmarz);
# println("Mixed = ",mixedFormulation);
# EHparam = ElasticHelmholtzParam(Mr,omega,2*m,m,m,gamma*m,false,mixedFormulation);
# McT = GetElasticHelmholtzOperator(EHparam);
# McT = McT/norm(McT,1);
# q = rand(spValType,size(McT,2));
# # q[100] = 1.0;
# x = zeros(spValType,size(McT,2));
# HKparam = getHybridCellWiseParam(McT,Mr,[4,4],0.5,numCores,innerKatzIter,mixedFormulation,Kaczmarz);

# McT = convert(SparseMatrixCSC{spValType,Int64},McT);
# prec = getHybridCellWisePrecond(HKparam,McT,copy(q),mixedFormulation,Kaczmarz);

# z = prec(q);
# println("Norm is: ",norm(McT'*z - q)/norm(q))

# y = zeros(spValType,size(McT,2));
# tic()
# y = FGMRES(McT,copy(q),y,6*inner,prec,1e-5*norm(q),true,true,numCores)[1];
# toc()
# println("Norm is: ",norm(McT'*y - q)/norm(q))
# y = zeros(spValType,size(McT,2));
# Afun = getAfun(McT,copy(y),numCores);
# x = KrylovMethods.fgmres(Afun,q,inner;tol = 1e-5,maxIter = 6,M = prec,x = y)[1];
# println("Norm is: ",norm(McT'*y - q)/norm(q))

