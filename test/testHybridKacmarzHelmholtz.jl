using SparseArrays
using LinearAlgebra
using jInv.Mesh
using Multigrid
using Multigrid.DomainDecomposition
using Helmholtz


domain = [0.0, 1.0, 0.0, 1.0];
n = [64,64];
Mr = getRegularMesh(domain,n)

pad = 8;
m = ones(tuple(n.+1...));

maxOmega = getMaximalFrequency(m,Mr);
ABLamp = maxOmega;

w = 0.9*maxOmega;
println("omega*h:");
println(w*Mr.h*sqrt(maximum(m)));
pad = pad*ones(Int64,Mr.dim);


println("************************************************* Hybrid Keczmarz for Acoustic Helmholtz 2D ******************************************************");

Sommerfeld = true;
NeumannAtFirstDim = true;
H,gamma = GetHelmholtzOperator(Mr,m,w,w*ones(size(m))*0.01,NeumannAtFirstDim,pad,ABLamp,Sommerfeld);
shift = 0.5;
SHT = sparse((H .+ GetHelmholtzShiftOP(m, real(w),shift))');


numCores = 2;
innerKatzIter = 3;
inner = 3;


q = getAcousticPointSource(Mr,ComplexF64)[1];
x = zeros(ComplexF64,size(q));
N = size(SHT,2);

HKparam = getHybridKaczmarz(ComplexF64,Int64, SHT,Mr,[4,4], getNodalIndicesOfCell,0.95,numCores,innerKatzIter);
prec = getHybridKaczmarzPrecond(HKparam,SHT);
x[:] .= FGMRES_relaxation(SHT,q[:],x[:],inner,prec,1e-3,true,true,numCores)[1]
println("Relative norm is: ",norm(SHT'*x[:] - q[:])./norm(q[:]))


println("In float precision:")

x[:].=0.0;
x = convert(Array{ComplexF32},x);
q = convert(Array{ComplexF32},q);
SHT = convert(SparseMatrixCSC{ComplexF32,Int64},SHT);

HKparam = getHybridKaczmarz(ComplexF32,Int64, SHT,Mr,[4,4], getNodalIndicesOfCell,0.95,numCores,innerKatzIter);
prec = getHybridKaczmarzPrecond(HKparam,SHT);
x[:] .= FGMRES_relaxation(SHT,q[:],x[:],inner,prec,1e-3,true,true,numCores)[1]
println("Relative norm is: ",norm(SHT'*x[:] - q[:])./norm(q[:]))
