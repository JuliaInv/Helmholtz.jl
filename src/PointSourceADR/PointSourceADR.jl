module PointSourceADR

using FactoredEikonalFastMarching;
using Helmholtz;
using jInv.Mesh;
using Multigrid;
using LinearAlgebra
using SparseArrays

using jInv.LinearSolvers
import jInv.Utils.clear!
import jInv.LinearSolvers.AbstractSolver
import jInv.LinearSolvers.solveLinearSystem


export getHelmholtzADR,rescaleMatrix,getADRcoefficients,restrictADRParam,ADRparam,getADRparam


mutable struct ADRparam
	Minv
	gamma
	m
	w
	T
	G
	LT
	src
	NeumannAtFirstDim
	Sommerfeld
	scaling
end

function getADRparam(Minv,gamma,m,w,T,G,LT,src,NeumannAtFirstDim,Somm,scaling=[])
	return ADRparam(Minv,gamma,m,w,T,G,LT,src,NeumannAtFirstDim,Somm,scaling);
end

"""
This function assumes that mesh_fine has a size dividable by 2 in cells.
"""
function restrictADRParam(mesh_fine,mesh_coarse,param_fine,level)
	gammac = restrictNodalVariables(param_fine.gamma,mesh_fine.n+1);
	mc = restrictNodalVariables(param_fine.m,mesh_fine.n+1);
	Tc = restrictNodalVariables(param_fine.T,mesh_fine.n+1);
	LTc = restrictNodalVariables(param_fine.LT,mesh_fine.n+1);
	Gc = Array(Array{Float64},mesh_fine.dim)
	for k=1:mesh_fine.dim
		Gc[k] = restrictNodalVariables(param_fine.G[k],mesh_fine.n+1);
	end
	src_c = div(param_fine.src-1,2)+1;
	scaling_c = [];
	if param_fine.scaling != []
		scaling_c = restrictNodalVariables(param_fine.scaling,mesh_fine.n+1);
	end
	param_coarse = ADRparam(mesh_coarse,gammac,mc,param_fine.w,Tc,Gc,LTc,src_c,param_fine.NeumannAtFirstDim,param_fine.Sommerfeld,scaling_c);
	return param_coarse;
end

function rescaleMatrix(AT::SparseMatrixCSC,M::Array)
# this function performs(spdiagm(M)*(AT)*spdiagm(1./M));
for ii = 1:size(AT,2)
	@inbounds invM = 1/M[ii];
	for j_idx = AT.colptr[ii]:AT.colptr[ii+1]-1
		@inbounds AT.nzval[j_idx] .*= (M[AT.rowval[j_idx]]*invM);
	end
end
end


function getADRcoefficients(Mesh::RegularMesh, mNodal::Array{Float64},src::Array{Int64})					
n_nodes = Mesh.n+1;

mem = getEikonalTempMemory(n_nodes);
eikParam = getEikonalParam(Mesh,mNodal,src,true);
println("TIME EIKONAL:")
tic()
solveFastMarchingUpwindGrad(eikParam, mem);
toc()
mem = 0;
							
LapNoBC = getNodalLaplacianMatrixNoBC(Mesh);
G = Array{Array{Float64}}(Mesh.dim);
T1 = eikParam.T1[:]; eikParam = 0;
L1 = LapNoBC*T1;
if Mesh.dim==2
	(T0,G01,G02,L0) = getAnalytic2DeikonalSolutionAll(n_nodes,Mesh.h,src);
	G01[src[1],src[2]] = 0.0;
	G02[src[1],src[2]] = 0.0;
	Dx1_long,Dx2_long = getNodalLongDiffGradientMatrix(Mesh);
	T0 = T0[:]; G01 = G01[:];G02 = G02[:];L0 = L0[:]; 
	G11 = Dx1_long*T1;
	G12 = Dx2_long*T1;
	G1 = T0.*G11 + T1.*G01
	G2 = T0.*G12 + T1.*G02
	G[1] = G1;
	G[2] = G2;
	T = T1.*T0;
	LT = L0.*T1  + 2.0.*(G11.*G01 + G12.*G02) + T0.*L1;
else
	(T0,G01,G02,G03,L0) = getAnalytic3DeikonalSolutionAll(n_nodes,Mesh.h,src);
	T0 = T0[:]; L0 = L0[:];
	T = T1.*T0;
	LT = LapNoBC*T[:];  ### WHY THIS IS NOT IDENTICAL TO BEFORE?????
	LapNoBC = 0;
	
	G01[src[1],src[2],src[3]] = 0.0;
	G02[src[1],src[2],src[3]] = 0.0;
	G03[src[1],src[2],src[3]] = 0.0;
	G01 = G01[:];G02 = G02[:];G03 = G03[:];
	Dx1_long,Dx2_long,Dx3_long = getNodalLongDiffGradientMatrix(Mesh);
	G11 = Dx1_long*T1;
	G12 = Dx2_long*T1;
	G13 = Dx3_long*T1;
	G1 = T0.*G11 + T1.*G01
	G2 = T0.*G12 + T1.*G02
	G3 = T0.*G13 + T1.*G03
	G[1] = G1;
	G[2] = G2;
	G[3] = G3;
	
	# LT = L0.*T1  + 2.0.*(G11.*G01 + G12.*G02 + G13.*G03) + T0.*L1;
	
end
return T,G,LT
end



function getHelmholtzADR(central::Bool, Mesh::RegularMesh, mNodal::Array{Float64}, omega::Float64, gamma::Array,
									NeumannAtFirstDim::Bool,src::Array{Int64},Sommerfeld::Bool,single::Bool = false)
T,G,LT = getADRcoefficients(Mesh, mNodal,src);
return getHelmholtzADR(central, Mesh, mNodal,omega, gamma,T,G,LT,NeumannAtFirstDim,src,Sommerfeld,single);
end			
									
function getHelmholtzADR(central::Bool, Mesh::RegularMesh, mNodal::Array{Float64},omega::Float64, gamma::Array,T,G,LT,
									NeumannAtFirstDim::Bool,src::Array{Int64},Sommerfeld::Bool,single::Bool = false)
## central = true : central + first
## central = false : upwind + 1st.
									
# mNodals = copy(mNodal);
# for k=1:64
	# for j = 2:size(mNodal,2)-1
		# for i = 2:size(mNodal,1)-1
			# @inbounds mNodals[i,j] = (2*mNodals[i,j] + (mNodals[i-1,j-1]+mNodals[i-1,j]+mNodals[i-1,j+1]+mNodals[i,j-1]+mNodals[i,j+1]+mNodals[i+1,j-1]+mNodals[i+1,j]+mNodals[i,j+1]))/10.0;
		# end
	# end
# end
n_nodes = Mesh.n.+1;
					
N = prod(n_nodes);
bcTauGrad = spzeros(N,N);
h = Mesh.h;
Som = zeros(size(gamma));
if Sommerfeld==true
	Som = getSommerfeldBC(Mesh,mNodal[:], omega,NeumannAtFirstDim);
end
n_tup = tuple(n_nodes...);
OPmap = zeros(Int8,n_tup);

if Mesh.dim==2
	# Dx1_long,Dx2_long = getNodalLongDiffGradientMatrixNeumann(Mesh);
	# bcTauGrad = getNeumannFromTauForGradDotGradTau(omega,G);
	# println(size(G[1]))
	# println(size(G[2]))
	# println(size(mNodal))
	
	REAC_EIK = (omega^2)*(G[1].^2 + G[2].^2 - mNodal);
	src_cs = Helmholtz.loc2cs(src,n_nodes);
	REAC_EIK[src_cs] = 0.0;
	
	if central == false
		Dx1_long = 0.0; Dx2_long = 0.0;
		OPmap[:] .= 2;
		OPmap[max(src[1]-1,1):min(src[1]+1,n_nodes[1]),max(src[2]-1,1):min(src[2]+1,n_nodes[2])] .= 0;
		ADV2 = 1im*2.0*omega*generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes);
	else
		OPmap[:] .= 0;
		ADV2 = 1im*2.0*omega*(G[1].*Dx1_long + G[2].*Dx2_long);
		Dx1_long = 0.0; Dx2_long = 0.0;
		# ADV2 = 1im*2.0*omega*generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes);
	end
	OPmap[:] .= 1;
	# OPmap[max(src[1]-1,1):min(src[1]+1,n_nodes[1]),max(src[2]-1,1):min(src[2]+1,n_nodes[2])] = 0;
else
	# Dx1_long,Dx2_long = getNodalLongDiffGradientMatrixNeumann(Mesh);
	# bcTauGrad = getNeumannFromTauForGradDotGradTau(omega,G);
	
	REAC_EIK = (omega^2)*(G[1].^2 + G[2].^2 + G[3].^2 - mNodal);
	src_cs = loc2cs3D(src,n_nodes);
	REAC_EIK[src_cs] = 0.0;
	# REAC_EIK[:] = 0.0;
	if central == false
		Dx1_long = 0.0; Dx2_long = 0.0;Dx3_long = 0.0;
		OPmap[:] .= 2;
		OPmap[max(src[1]-1,1):min(src[1]+1,n_nodes[1]),max(src[2]-1,1):min(src[2]+1,n_nodes[2]),max(src[3]-1,1):min(src[3]+1,n_nodes[3])] = 0;
		ADV2 = 1im*2.0*omega*generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes);
		
	else
		OPmap[:] .= 0;
		
		# ADV2 = 1im*2.0*omega*(G[1].*Dx1_long + G[2].*Dx2_long + G[3].*Dx3_long);
		Dx1_long = 0.0; Dx2_long = 0.0;Dx3_long = 0.0;
		# ADV2 = 1im*2.0*omega*generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes);
		ADV2 = 1im*1.0*omega*(G1.*Dx1_long + G2.*Dx2_long + G3.*Dx3_long + Dx1_long*spdiagm(G1) + Dx2_long*spdiagm(G2) + Dx3_long*spdiagm(G3)); LT[:] = 0.0;
		
		
	end
	
	OPmap[:] .= 1;
	# OPmap[max(src[1]-1,1):min(src[1]+1,n_nodes[1]),max(src[2]-1,1):min(src[2]+1,n_nodes[2]),max(src[3]-1,1):min(src[3]+1,n_nodes[3])] = 0;
end

# v1 = spdiagm(REAC_EIK + 1im*omega*(gamma[:].*mNodal[:]) -  Som[:]);
# println(sizeof(LT[:]))
# println(sizeof(getNodalAveragingMatrix(Mesh)))

# NA = LT[:].*getNodalAveragingMatrix(Mesh)
# REAC = v1 + 1im*omega*NA;
bcTauLap = getNeumannFromTauForLap(omega,G,h,n_tup);
REAC = spdiagm(0=>(REAC_EIK[:] + 1im*omega*(LT[:]+gamma[:].*mNodal[:]) -  Som[:]));

##########################################################################################
#### backup code for no laplacian of tau #################################################
# ADV_longUp = generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes);
# OPmap[max(src[1]-1,1):min(src[1]+1,n_nodes[1]),max(src[2]-1,1):min(src[2]+1,n_nodes[2])] = 0;
# ADV_longUp += generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes,true);
# ADV_longUp = 1im*omega*(ADV_longUp);
# OPmap[:]=1;
# ADV_shortUp = generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes);
# ADV_shortUp += generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes,true);
# ADV_shortUp = 1im*omega*(ADV_longUp)
##########################################################################################
##########################################################################################
	
	if single 
		ADV2 = convert(SparseMatrixCSC{ComplexF32,Int32},ADV2);
	end
	ADRtemp = Helmholtz.getNodalLaplacianMatrix(Mesh) + bcTauLap + REAC + bcTauGrad;REAC = 0;bcTauLap = 0;
	
	if single 
	 ADRtemp = convert(SparseMatrixCSC{ComplexF32,Int32},ADRtemp);
	end
	ADR2 = ADRtemp + ADV2;ADV2 = 0;
	# if single 
	 # ADR2 = convert(SparseMatrixCSC{Complex64,Int32},ADR2);
	# end
	ADV1 = 1im*2.0*omega*generateSecondOrderUpwindAdvection(OPmap,G,h,n_nodes);
	if single 
	 ADV1 = convert(SparseMatrixCSC{ComplexF32,Int32},ADV1);
	end
	ADR1 = ADRtemp + ADV1; ADRtemp = 0;
	return ADR2,ADR1,T;
end

function ddxLong(n,h)
# D = ddx(n), 1D long difference derivative operator [-1 0 1]/2
	D = spdiagm((fill(-0.5/h,n),fill(0.5/h,n)),(-1,1),n+1,n+1)
	D[1,1:2] = [-1.0 1.0]./h;
	D[end,end-1:end] = [-1.0 1.0]./h;
	return D;
end 

function getNodalLongDiffGradientMatrix(Mesh::RegularMesh)
if Mesh.dim==2
	G1 = kron(speye(Mesh.n[2]+1),ddxLong(Mesh.n[1],Mesh.h[1]))
	G2 = kron(ddxLong(Mesh.n[2],Mesh.h[2]),speye(Mesh.n[1]+1))
	return (G1,G2);
elseif Mesh.dim==3
	G1 = kron(speye(Mesh.n[3]+1),kron(speye(Mesh.n[2]+1),ddxLong(Mesh.n[1],Mesh.h[1])))
	G2 = kron(speye(Mesh.n[3]+1),kron(ddxLong(Mesh.n[2],Mesh.h[2]),speye(Mesh.n[1]+1)))	
	G3 = kron(ddxLong(Mesh.n[3],Mesh.h[3]),kron(speye(Mesh.n[2]+1),speye(Mesh.n[1]+1)))
	return (G1,G2,G3);
end
end


function getNeumannFromTauForLap(omega::Float64,G::Array{Array{Float64},1},h::Array{Float64},n_tup)
if length(G)==2
	bcTau1 = zeros(ComplexF64,n_tup);
	bcTau2 = zeros(ComplexF64,n_tup);
	# for 1: minus because normal is pointing -x but we use exp(-i*w*tau) so its plus
	# for end: plus because normal is pointing +x but we use exp(-i*w*tau) so its minus
	G1 = reshape(G[1],n_tup);
	G2 = reshape(G[2],n_tup);
	bcTau1[1,:]   += +1im*(BC()/h[1])*omega*G1[1,:];
	bcTau1[end,:] += -1im*(BC()/h[1])*omega*G1[end,:];
	bcTau2[:,1]   += +1im*(BC()/h[2])*omega*G2[:,1];
	bcTau2[:,end] += -1im*(BC()/h[2])*omega*G2[:,end];
	bcTau = spdiagm(0=>vec(bcTau1 + bcTau2));
	return bcTau
else
	bcTau1 = zeros(ComplexF64,n_tup);
	bcTau2 = zeros(ComplexF64,n_tup);
	bcTau3 = zeros(ComplexF64,n_tup);
	G1 = reshape(G[1],n_tup);
	G2 = reshape(G[2],n_tup);
	G3 = reshape(G[3],n_tup);
	bcTau1[1,:,:]   += +1im*(BC()/h[1])*omega*G1[1,:,:];
	bcTau1[end,:,:] += -1im*(BC()/h[1])*omega*G1[end,:,:];
	bcTau2[:,1,:]   += +1im*(BC()/h[2])*omega*G2[:,1,:];
	bcTau2[:,end,:] += -1im*(BC()/h[2])*omega*G2[:,end,:];
	bcTau3[:,:,1]   += +1im*(BC()/h[3])*omega*G3[:,:,1];
	bcTau3[:,:,end] += -1im*(BC()/h[3])*omega*G3[:,:,end];
	bcTau = spdiagm(0=>vec(bcTau1 + bcTau2 + bcTau3));
	return bcTau
end
end

function av(n)
p = zeros(n+1);
p[1] = 0.5;
p[end] = 0.5;
A = spdiagm((fill(0.5,n),p,fill(0.5,n)),(-1,0,1),n+1,n+1)
end
function getNodalAveragingMatrix(Mesh::RegularMesh)
if Mesh.dim==2
	G1 = kron(speye(Mesh.n[2]+1),av(Mesh.n[1]))
	G2 = kron(av(Mesh.n[2]),speye(Mesh.n[1]+1))
	return 0.5*(G1+G2);
elseif Mesh.dim==3
	G1 = kron(speye(Mesh.n[3]+1),kron(speye(Mesh.n[2]+1),av(Mesh.n[1])))
	G2 = kron(speye(Mesh.n[3]+1),kron(av(Mesh.n[2]),speye(Mesh.n[1]+1)))	
	G3 = kron(av(Mesh.n[3]),kron(speye(Mesh.n[2]+1),speye(Mesh.n[1]+1)))
	return (G1+G2+G3)/3.0;
end
end




function generateSecondOrderUpwindAdvection(OPmap::Array{Int8},G::Array{Array{Float64},1},h::Array{Float64,1},n::Array{Int64},flipAdvectionVector::Bool=false)
N = prod(n);
invh = 1.0./h;
nnzidx = 1;
dim = length(n);
II = ones(Int32,N*(3*dim));
JJ = ones(Int32,length(II));
VV = zeros(Float32,length(II));
invh = 1.0./h;
idxnnz = 1;
offset = [1;n[1];n[1]*n[2]];
for d = 1:dim
	OPmapd = copy(OPmap);
	if dim==2
		if d==1
			OPmapd[1:2,:] .= 1;
			OPmapd[end-1:end,:] .= 1;
		else
			OPmapd[:,1:2] .= 1;
			OPmapd[:,end-1:end] .= 1;
		end
	else
		if d==1
			OPmapd[1:2,:,:] .= 1;
			OPmapd[end-1:end,:,:] .= 1;
		elseif d==2
			OPmapd[:,1:2,:] .= 1;
			OPmapd[:,end-1:end,:] .= 1;
		else
			OPmapd[:,:,1:2] .= 1;
			OPmapd[:,:,end-1:end] .= 1;
		end
	end
	for ii = 1:N
		@inbounds II[idxnnz:(idxnnz+3 - 1)] .= ii;
		if flipAdvectionVector==false
			updateUpwindStencil(G[d][ii],OPmapd[ii],JJ,VV,invh[d],ii,offset[d],idxnnz);
		else
			updateUpwindStencilFlipped(G[d],OPmapd[ii],JJ,VV,invh[d],ii,offset[d],idxnnz);
		end
		idxnnz += 3;
	end
end
JJ[JJ.<1] .= 1;
JJ[JJ.>N] .= 1;
ADV = sparse(II,JJ,VV,N,N);
return ADV;
end

# function generateSecondOrderUpwindGrad(OPmap::Array{Int8},G::Array{Array{Float64},1},h::Array{Float64,1},n::Array{Int64},flipAdvectionVector::Bool=false)
# N = prod(n);
# invh = 1./h;
# nnzidx = 1;
# dim = length(n);
# II = ones(Int64,N*(3*dim));
# JJ = ones(Int64,length(II));
# VV = zeros(Float64,length(II));
# invh = 1./h;
# idxnnz = 1;
# offset = [1;n[1];n[1]*n[2]];
# for ii = 1:N
	# II[idxnnz:(idxnnz+3*dim - 1)] = ii;
	# for d = 1:dim
		# if flipAdvectionVector==false
			# updateUpwindStencil(G[d][ii],OPmap[ii],JJ,VV,invh[d],ii,offset[d],idxnnz);
		# else
			# updateUpwindStencilFlipped(G[d],OPmap[ii],JJ,VV,invh[d],ii,offset[d],idxnnz);
		# end
		# idxnnz += 3;
	# end
# end
# # JJ[JJ.<1] = 1;
# # JJ[JJ.>N] = 1;
# ADV = sparse(II,JJ,VV,N,N);
# return ADV;
# end


function updateUpwindStencil(a::Float64,op::Int8,ansidx::Union{Array{Int32},Array{Int64}},ansval::Union{Array{Float32},Array{Float64}},invh::Float64,i::Int64,offset::Int64,idxnnz::Int64)
	if a > 0.0
		if op == 2
			@inbounds ansval[idxnnz] = a*1.5*invh; 
			@inbounds ansval[idxnnz+1] = -a*2.0*invh;
			@inbounds ansval[idxnnz+2] = a*0.5*invh;
			@inbounds ansidx[idxnnz] 	 = i; 
			@inbounds ansidx[idxnnz+1] = i-offset;
			@inbounds ansidx[idxnnz+2] = i-2*offset;
		elseif op == 1 
			@inbounds ansval[idxnnz]   = a*invh; 
			@inbounds ansval[idxnnz+1] = -a*invh;
			@inbounds ansidx[idxnnz] 	 = i; 
			@inbounds ansidx[idxnnz+1] = i-offset;
		elseif op == 0
			@inbounds ansval[idxnnz]   = a*0.5*invh; 
			@inbounds ansval[idxnnz+1] = -a*0.5*invh;
			@inbounds ansidx[idxnnz] 	 = i+offset; 
			@inbounds ansidx[idxnnz+1] = i-offset;
		end
	elseif a < 0.0
		if op == 2
			@inbounds ansval[idxnnz]   = -a*1.5*invh; 
			@inbounds ansval[idxnnz+1] = a*2.0*invh;
			@inbounds ansval[idxnnz+2] = -a*0.5*invh;
			@inbounds ansidx[idxnnz] 	 = i; 
			@inbounds ansidx[idxnnz+1] = i+offset;
			@inbounds ansidx[idxnnz+2] = i+2*offset;
		elseif op==1
			@inbounds ansval[idxnnz]   = -a*invh; 
			@inbounds ansval[idxnnz+1] = a*invh;
			@inbounds ansidx[idxnnz] 	 = i; 
			@inbounds ansidx[idxnnz+1] = i+offset;
		elseif op==0
			@inbounds ansval[idxnnz]   = a*0.5*invh; 
			@inbounds ansval[idxnnz+1] = -a*0.5*invh;
			@inbounds ansidx[idxnnz] 	 = i+offset; 
			@inbounds ansidx[idxnnz+1] = i-offset;
		end
	end
end

function updateUpwindStencilFlipped(a::Array{Float64},op::Int8,ansidx::Union{Array{Int32},Array{Int64}},ansval::Union{Array{Float32},Array{Float64}},invh::Float64,i::Int64,offset::Int64,idxnnz::Int64)
	if a[i] > 0.0
		if op == 2
			ansval[idxnnz] = a[i]*1.5*invh; 
			ansval[idxnnz+1] = -a[i-offset]*2.0*invh;
			ansval[idxnnz+2] = a[i-2*offset]*0.5*invh;
			ansidx[idxnnz] 	 = i; 
			ansidx[idxnnz+1] = i-offset;
			ansidx[idxnnz+2] = i-2*offset;
		elseif op == 1 
			ansval[idxnnz]   = a[i]*invh; 
			ansval[idxnnz+1] = -a[i-offset]*invh;
			ansidx[idxnnz] 	 = i; 
			ansidx[idxnnz+1] = i-offset;
		elseif op == 0
			ansval[idxnnz]   = a[i+offset]*0.5*invh; 
			ansval[idxnnz+1] = -a[i-offset]*0.5*invh;
			ansidx[idxnnz] 	 = i+offset; 
			ansidx[idxnnz+1] = i-offset;
		end
	elseif a[i] <= 0.0
		if op == 2
			ansval[idxnnz]   = -a[i]*1.5*invh; 
			ansval[idxnnz+1] = a[i+offset]*2.0*invh;
			ansval[idxnnz+2] = -a[i+2*offset]*0.5*invh;
			ansidx[idxnnz] 	 = i; 
			ansidx[idxnnz+1] = i+offset;
			ansidx[idxnnz+2] = i+2*offset;
		elseif op==1
			ansval[idxnnz]   = -a[i]*invh; 
			ansval[idxnnz+1] = a[i+offset]*invh;
			ansidx[idxnnz] 	 = i; 
			ansidx[idxnnz+1] = i+offset;
		elseif op==0
			ansval[idxnnz]   = a[i+offset]*0.5*invh; 
			ansval[idxnnz+1] = -a[i-offset]*0.5*invh;
			ansidx[idxnnz] 	 = i+offset; 
			ansidx[idxnnz+1] = i-offset;
		end
	end
end



function dxxMatNoBC(n::Int64,h::Float64)
O0 = zeros(n-2);
O1 = ones(n-1);
O2 = -2.0*ones(n);
O3 = ones(n-1);
O4 = zeros(n-2);
O0[end] = 1.0;
O1[end] = -2.0;
O2[end] = 1.0;
O2[1] = 1.0;
O3[1] = -2.0;
O4[1] = 1.0;
dxx = spdiagm((O0/h^2,O1/(h^2),O2/(h^2),O3/(h^2),O4/(h^2)),[-2,-1,0,1,2],n,n);
return dxx;
end

function getNodalLaplacianMatrixNoBC(Msh::RegularMesh)
nodes = Msh.n+1;
I1 = speye(nodes[1]);
Dxx1 = dxxMatNoBC(nodes[1],Msh.h[1]);
I2 = speye(nodes[2]);
Dxx2 = dxxMatNoBC(nodes[2],Msh.h[2]);
Dxx3 = spzeros(0);
if Msh.dim==2
	L = kron(I2,Dxx1) + kron(Dxx2,I1);
else
	I3 = speye(nodes[3]);
	Dxx3 = dxxMatNoBC(nodes[3],Msh.h[3]);
	L = kron(I3,kron(I2,Dxx1) + kron(Dxx2,I1)) + kron(Dxx3,kron(I2,I1));
end
return L;
end
end

# function ddxLongNeumann(n,h)
	# D = spdiagm((fill(-0.5/h,n),fill(0.5/h,n)),(-1,1),n+1,n+1)
	# D[1,1:2] = 0.0;
	# D[end,end-1:end] = 0.0;
	# return D;
# end
# function getNodalLongDiffGradientMatrixNeumann(Mesh::RegularMesh)
# if Mesh.dim==2
	# G1 = kron(speye(Mesh.n[2]+1),ddxLongNeumann(Mesh.n[1],Mesh.h[1]))
	# G2 = kron(ddxLongNeumann(Mesh.n[2],Mesh.h[2]),speye(Mesh.n[1]+1))
	# return (G1,G2);
# elseif Mesh.dim==3
	# G1 = kron(speye(Mesh.n[3]+1),kron(speye(Mesh.n[2]+1),ddxLongNeumann(Mesh.n[1])))
	# G2 = kron(speye(Mesh.n[3]+1),kron(ddxLongNeumann(Mesh.n[2]),speye(Mesh.n[1]+1)))	
	# G3 = kron(ddxLongNeumann(Mesh.n[3]),kron(speye(Mesh.n[2]+1),speye(Mesh.n[1]+1)))
	# return (G1,G2,G3);
# end
# end

# function getNeumannFromTauForGradDotGradTau(omega::Float64,G::Array{Array{Float64},1})
#	error("Fix the bug here like in Neumann for lap")
# if length(G)==2
	# bcTau1 = zeros(Complex128,size(G[1]));
	# bcTau2 = zeros(Complex128,size(G[2]));
	# bcTau1[1,:]   += +1im*omega*G[1][1,:];
	# bcTau1[end,:] += -1im*omega*G[1][end,:];
	# bcTau2[:,1]   += +1im*omega*G[2][:,1];
	# bcTau2[:,end] += -1im*omega*G[2][:,end];
	# bcTau = spdiagm(2.0*1im*omega*(bcTau1.*G[1] + bcTau2.*G[2]));
# else
	# bcTau1 = zeros(Complex128,size(G[1]));
	# bcTau2 = zeros(Complex128,size(G[2]));
	# bcTau3 = zeros(Complex128,size(G[3]));
	# bcTau1[1,:,:]   += +1im*omega*G[1][1,:,:];
	# bcTau1[end,:,:] += -1im*omega*G[1][end,:,:];
	# bcTau2[:,1,:]   += +1im*omega*G[2][:,1,:];
	# bcTau2[:,end,:] += -1im*omega*G[2][:,end,:];
	# bcTau3[:,:,1]   += +1im*omega*G[3][:,:,1];
	# bcTau3[:,:,end] += -1im*omega*G[3][:,:,end];
	# bcTau = spdiagm(2.0*1im*omega*(bcTau1.*G[1] + bcTau2.*G[2] + bcTau3.*G[3]));
# end
# return bcTau
# end



