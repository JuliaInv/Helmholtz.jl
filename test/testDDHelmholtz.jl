using jInv.Mesh;
using Helmholtz
using Multigrid
using Multigrid.DomainDecomposition
using Multigrid.ParallelJuliaSolver
using SparseArrays
using LinearAlgebra
using jInv.LinearSolvers


const plotting = false;
if plotting
	using PyPlot;
	close("all")
end


# m = readdlm("SEGmodel2Dsalt.dat"); m = m';
#m = m*1e-3;

m = ones(513,257);

Minv = getRegularMesh([0.0,13.5,0.0,4.2],collect(size(m)) .- 1);

pad = 16;
pad = pad*ones(Int64,Minv.dim);

m = 1.0./m.^2

if plotting
	figure();
	imshow(1.0./sqrt.(Matrix(m')))
end

# omega = 0.2*pi / (maximum(h)*maximum(sqrt.(m)))
omega = getMaximalFrequency(m,Minv);
ABLamp = omega;
println("omega is ",omega/pi," times pi")
gamma_0 = 0.005
gamma = getABL(Minv.n.+1,false,ones(Int64,2).*pad,Float64(omega)) .+ gamma_0*omega

Hparam = HelmholtzParam(Minv,gamma,m,omega,false,true)
q,src = getAcousticPointSource(Minv,ComplexF64);
q = vec(q);

H = GetHelmholtzOperator(Minv,m,omega,ones(size(m))*gamma_0,true,pad,ABLamp,true)[1];
# H = GetHelmholtzOperator(Hparam)[1];
HrT = sparse(H')

#Shift = GetHelmholtzShiftOP(m, omega,0.1);
#Shift = convert(SparseMatrixCSC{ComplexF64,spIndType},Shift);


NumCells = [8,4];
overlap = [4,4];

# DDparam = getDomainDecompositionParam(ComplexF64,Int64,Minv,NumCells,overlap,getNodalIndicesOfCell,getParallelJuliaSolver(ComplexF64,Int64,numCores=4,backend=3));
DDparam = getDomainDecompositionParam(ComplexF64,Int64,Minv,NumCells,overlap,getNodalIndicesOfCell,getJuliaSolver());

println("Performing Absorbing+Neumann Setup")
println("NumCells = ",NumCells," overlapp = ",overlap);

function getSubParams(Hparam, M::RegularMesh,i::Array{Int64},NumCells::Array{Int64},Overlap::Array{Int64})
	subMesh   = getSubMeshOfCell(NumCells,overlap,i,M);
	IIp       = getNodalIndicesOfCell(NumCells,overlap,i,M.n);
	code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2]];
	subgamma  = getABL(subMesh.n.+1,i[end]==1,overlap.+4,2.0./(M.h[1]),code).+0.001*Hparam.omega;
    Hparam = HelmholtzParam(subMesh,Hparam.gamma[IIp]+subgamma[:],Hparam.m[IIp],Hparam.omega,false,true)
	return Hparam;
end


# getDDMass = (ddp,hp,i)->(0.0.*Vector(diag(GetHelmholtzShiftOP(hp.m,0.0,0.0))));

#function getDirichletMassNodalMesh(DDparam::DomainDecompositionParam,problem_param::HelmholtzParam,i::Array{Int64})
# 	d = getDirichletMassNodal(DDparam.numDomains,DDparam.overlap,i,DDparam.Mesh.n);
# 	d.*=(0.1*4.0)/prod(DDparam.Mesh.h);
# 	return d;
#end
# getDDMass = getDirichletMassNodalMesh

Ctor = DomainDecompositionOperatorConstructor{ComplexF64,Int64}(Hparam,getSubParams,GetHelmholtzOperator,identity);
DDparam = setupDDSerial(Ctor,DDparam);
println("Performing DD Solution with GMRES")
x = copy(q); 
x[:] .= 0.0;


x = solveLinearSystem!(HrT,q,x,DDparam)[1];
println("Outside error: ", norm(HrT'*x - q)/norm(q))

# x, = solveDDSerial(HrT,q,zeros(ComplexF64,size(q)),DDparam,20);
# println("Outside error: ", norm(HrT'*x - q)/norm(q))