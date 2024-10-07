
export GetHelmholtzOperator,GetHelmholtzOperatorHO,GetHelmholtzShiftOP,getABL,getSommerfeldBC,getHelmholtzFun,getMaximalFrequency

"""
function  GetHelmholtzOperator

Constructor for the acoustic Helmholtz equation

Special Fields:	
omega can be real or complex for artificial attenuation. 
Artificial attenuation through a complex omega keeps the phase fixed.
gamma - True attenuation parameter, corresponding to the time domain equation:  u_tt + gamma*ut + c^2*L*u = q
"""
function GetHelmholtzOperator(Hparam::HelmholtzParam,orderNeumannBC::Int64 = 2)
	H = GetHelmholtzOperator(Hparam.Mesh, Hparam.m, Hparam.omega, Hparam.gamma, Hparam.NeumannOnTop,Hparam.Sommerfeld,orderNeumannBC);
end

function GetHelmholtzOperatorHO(Hparam::HelmholtzParam)
	H = GetHelmholtzOperatorHO(Hparam.Mesh, Hparam.m, Hparam.omega, Hparam.gamma, Hparam.NeumannOnTop,Hparam.Sommerfeld);
end

function GetHelmholtzOperator(Msh::RegularMesh, mNodal::Array{Float64}, omega::Union{Float64,ComplexF64}, gamma::Array,
									NeumannAtFirstDim::Bool,ABLpad::Array{Int64},ABLamp::Float64,Sommerfeld::Bool,orderNeumannBC::Int64 = 2)
if gamma == []
	gamma = getABL(Msh.n.+1,NeumannAtFirstDim,ABLpad,ABLamp);
else
	gamma += getABL(Msh.n.+1,NeumannAtFirstDim,ABLpad,ABLamp);
end
H = GetHelmholtzOperator(Msh, mNodal, omega, gamma, NeumannAtFirstDim,Sommerfeld,orderNeumannBC);
return H,gamma
end

function GetHelmholtzOperator(Msh::RegularMesh, mNodal::Array{Float64}, omega::Union{Float64,ComplexF64}, gamma::Array{Float64},
							  NeumannAtFirstDim::Bool,Sommerfeld::Bool,orderNeumannBC::Int64 = 2)
Lap   = getNodalLaplacianMatrix(Msh,orderNeumannBC);
# this code computes a laplacian the long way, AND stores the gradient on Msh... So we avoid using it.
# Grad  = getNodalGradientMatrix(Msh) 
# Lap   = Grad'*Grad

# mass = -((omega.^2).*(mNodal[:]).*(1.0.+1im*gamma[:]));
mass = -(omega.^2).*(mNodal[:]).*(1.0.-1im*gamma[:]./real(omega));

if Sommerfeld
	# println("Adding Sommerfeld");
	somm = getSommerfeldBC(Msh,mNodal,real(omega),NeumannAtFirstDim);
	mass -= somm[:];
end
H = Lap .+ sparse(Diagonal(mass));
return H;
end



function GetHelmholtzOperatorHO(Msh::RegularMesh, mNodal::Array{Float64}, omega::Union{Float64,ComplexF64}, gamma::Array{Float64},
							  NeumannAtFirstDim::Bool,Sommerfeld::Bool,beta = 1.0)
# Lap   = getNodalLaplacianMatrix(Msh,orderNeumannBC);
Lap,M = getSpreadNodalLaplacianAndMass(Msh,beta)
# this code computes a laplacian the long way, AND stores the gradient on Msh... So we avoid using it.
# Grad  = getNodalGradientMatrix(Msh) 
# Lap   = Grad'*Grad

# mass = -((omega.^2).*(mNodal[:]).*(1.0.+1im*gamma[:]));
mass = -(omega.^2).*(mNodal[:]).*(1.0.-1im*gamma[:]./real(omega));

if Sommerfeld
	# println("Adding Sommerfeld");
	somm = getSommerfeldBC(Msh,mNodal,real(omega),NeumannAtFirstDim);
	mass -= somm[:];
end
H = Lap .+ M*sparse(Diagonal(mass));
return H;
end


function getMaximalFrequency(m::Union{Array{Float64},Array{Float32},Float64},M::RegularMesh)
## m here is in slowness squared
omegamax = (0.1*2*pi)./(maximum(M.h)*sqrt(maximum(m)));
return omegamax;
end

function GetHelmholtzShiftOP(mNodal::Array{Float64}, omega::Float64,shift::Float64)
return sparse(Diagonal(mNodal[:].*(1im*shift*omega^2)));
end

function getHelmholtzFun(ShiftedHelmholtT::SparseMatrixCSC,ShiftMat::SparseMatrixCSC,y::ArrayTypes,numCores::Int64)
function Hfun(x)
		# # here we avoid the storage of the Helmholtz matrix by using the shifted matrix plus the shift.
		Zero = zero(ComplexF64);
		One = one(ComplexF64);
		SpMatMul(One,ShiftedHelmholtT,x,Zero,y,numCores);
		SpMatMul(One,ShiftMat,x,One,y,numCores);
		return y;
end
return Hfun;
end

function getABL(n::Array{Int64},NeumannAtFirstDim::Bool,ABLpad::Array{Int64},ABLamp::Float64,code=ones(Bool,length(n),2))
  pad = ABLpad;
  ntup = tuple(n...);
  impl = 1;
  if length(n)==2
	if impl == 0
		x1 = range(-1,stop=1,length=n[1]);
		x2 = range(-1,stop=1,length=n[2]);
		X1,X2 = ndgrid(x1,x2);
		padx1 = ABLpad[1];
		padx2 = ABLpad[2];
		
		gammax = zeros(size(X1));
		if code[1,1] 
			gammaxL = (X1 .- x1[padx1]).^2;
			gammaxL[padx1+1:end,:] .= 0
			gammax.+=gammaxL;
		end
		if code[1,2] 
			gammaxR = (X1 .- x1[end-padx1+1]).^2
			gammaxR[1:end-padx1,:] .= 0
			gammax.+=gammaxR;
		end
		gammax ./= (maximum(gammax)+1e-5);
		
		if NeumannAtFirstDim==true
			code[2,1] = false;
		end
		gammaz = zeros(size(X1));
		if code[2,1] 
			gammaz1 = (X2 .- x2[padx2]).^2;
			gammaz1[:,padx2+1:end] .= 0
			gammaz.+=gammaz1;
		end
		if code[2,2]
			gammaz2 = (X2 .- x2[end-padx2+1]).^2
			gammaz2[:,1:end-padx2] .= 0
			gammaz.+=gammaz2;
		end
		gammaz ./= (maximum(gammaz)+1e-5);
	
		gamma = gammax .+ gammaz
		gamma .*= ABLamp;
		gamma[gamma.>=ABLamp] .= ABLamp;
	else
		gamma = zeros(ntup);
		b_bwd1 = ((pad[1]:-1:1).^2)./pad[1]^2;
		b_bwd2 = ((pad[2]:-1:1).^2)./pad[2]^2;
  
		b_fwd1 = ((1:pad[1]).^2)./pad[1]^2;
		b_fwd2 = ((1:pad[2]).^2)./pad[2]^2;
		I1 = (n[1] - pad[1] + 1):n[1];
		I2 = (n[2] - pad[2] + 1):n[2];
  
		if NeumannAtFirstDim==false
			gamma[:,1:pad[2]] += ones(n[1],1)*b_bwd2';
			gamma[1:pad[1],1:pad[2]] -= b_bwd1*b_bwd2';
			gamma[I1,1:pad[2]] -= b_fwd1*b_bwd2';
		end

		gamma[:,I2] +=  ones(n[1],1)*b_fwd2';
		gamma[1:pad[1],:] += b_bwd1*ones(1,n[2]);
		gamma[I1,:] += b_fwd1*ones(1,n[2]);
		gamma[1:pad[1],I2] -= b_bwd1*b_fwd2';
		gamma[I1,I2] -= b_fwd1*b_fwd2';
		gamma *= ABLamp;
	end
  else
  
	x1 = range(-1,stop=1,length=n[1]);
	x2 = range(-1,stop=1,length=n[2]);
	x3 = range(0,stop=1,length=n[3]);
	X1,X2,X3 = ndgrid(x1,x2,x3);
	padx1 = ABLpad[1];
	padx2 = ABLpad[2];
	padx3 = ABLpad[3];
	gammax = zeros(size(X1));
	if code[1,1] 
		gammaL = (X1 .- x1[padx1]).^2;
		gammaL[padx1+1:end,:,:] .= 0.0
		gammax.+=gammaL;
	end
	if code[1,2] 
		gammaR = (X1 .- x1[end-padx1+1]).^2
		gammaR[1:end-padx1,:,:] .= 0.0
		gammax.+=gammaR;
	end
	gammax ./= (maximum(gammax)+1e-5);
	
	gammay = zeros(size(X2));
	if code[2,1] 
		gammaL = (X2 .- x2[padx2]).^2;
		gammaL[:,padx2+1:end,:] .= 0.0
		gammay.+=gammaL;
	end
	if code[2,2] 
		gammaR = (X2 .- x2[end-padx2+1]).^2
		gammaR[:,1:end-padx2,:] .= 0.0
		gammay.+=gammaR;
	end
	gammay ./= (maximum(gammay)+1e-5);
	
	if NeumannAtFirstDim==true
		code[3,1] = false;
	end
	gammaz = zeros(size(X3));
	if code[3,1] 
		gammaL = (X3 .- x3[padx3]).^2;
		gammaL[:,:,padx3+1:end] .= 0.0
		gammaz.+=gammaL;
	end
	if code[3,2] 
		gammaR = (X3 .- x3[end-padx3+1]).^2
		gammaR[:,:,1:end-padx3] .= 0.0
		gammaz.+=gammaR;
	end
	gammaz ./= (maximum(gammaz)+1e-5);
	
	gamma  = gammax + gammay + gammaz;
	gamma .*= ABLamp;
	gamma[gamma.>=ABLamp] .= ABLamp;
  end
  return gamma;
end

function getSommerfeldBC(Msh::RegularMesh,mNodal::Array{Float64}, omega::Float64,NeumannOnTop::Bool,orderNeumannBC=2)
BC = getBC(orderNeumannBC);
ntup = tuple((Msh.n .+ 1)...);
Somm = zeros(ComplexF64,ntup);
mNodal = reshape(mNodal,ntup);
h = Msh.h;

if Msh.dim==2
	if !NeumannOnTop
		Somm[:,1] += -1im*omega*(BC/h[2]).*sqrt.(mNodal[:,1]);
	end
	Somm[:,end] += (-1im*omega*(BC/h[2])).*sqrt.(mNodal[1:end,end]);
	Somm[end,:] += (-1im*omega*(BC/h[1])).*sqrt.(mNodal[end,:]);
	Somm[1,:] += (-1im*omega*(BC/h[1])).*sqrt.(mNodal[1,:]);
else
	if !NeumannOnTop
		Somm[:,:,1] += -1im*omega*(BC/h[3]).*sqrt.(mNodal[:,:,1]);
	end
	Somm[:,:,end] += -1im*omega*(BC/h[3]).*sqrt.(mNodal[:,:,end]);
	Somm[:,1,:] += -1im*omega*(BC/h[2]).*sqrt.(mNodal[:,1,:]);
	Somm[:,end,:] += -1im*omega*(BC/h[2]).*sqrt.(mNodal[:,end,:]);
	Somm[1,:,:] += -1im*omega*(BC/h[1]).*sqrt.(mNodal[1,:,:]);
	Somm[end,:,:] += -1im*omega*(BC/h[1]).*sqrt.(mNodal[end,:,:]);
end
return Somm;
end