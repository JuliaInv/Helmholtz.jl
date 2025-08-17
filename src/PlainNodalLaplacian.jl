export getNodalLaplacianMatrix, multOpNeumann!, Lap2DStencil,dxxMat


function getBC(orderNeumannBC)
BC = 2.0;
if orderNeumannBC == 2
	#one is for first order Neumann, and 2 is for 2nd order. Note that with 2, the matrix is not symmetric!!!
	BC = 2.0;
elseif orderNeumannBC == 1
	BC = 1.0;
else
	println("getNodalLaplacianMatrix: BC not supported")
end
return BC;
end


function dxxMat(n::Int64,h::Float64,orderNeumannBC=2)

BC = getBC(orderNeumannBC);
O1 = -ones(n-1);
O1[n-1] = -BC;
O2 = 2.0*ones(n);
O2[1] = BC;
O2[n] = BC;
O3 = -ones(n-1);
O3[1] = -BC;
dxx = spdiagm(-1 => O1./(h^2), 0 => O2./(h^2), 1=> O3./(h^2))
return dxx;
end

function getNodalLaplacianMatrix(Msh::RegularMesh,orderNeumannBC=2)
nodes = Msh.n .+ 1;
I1 = SparseMatrixCSC(1.0I, nodes[1], nodes[1]);
Dxx1 = dxxMat(nodes[1],Msh.h[1],orderNeumannBC);
I2 = SparseMatrixCSC(1.0I, nodes[2], nodes[2]);
Dxx2 = dxxMat(nodes[2],Msh.h[2],orderNeumannBC);
if Msh.dim==2
	L = kron(I2,Dxx1) + kron(Dxx2,I1);
else
	I3 = SparseMatrixCSC(1.0I, nodes[3], nodes[3]);
	Dxx3 = dxxMat(nodes[3],Msh.h[3],orderNeumannBC);
	L = kron(I3,kron(I2,Dxx1) .+ kron(Dxx2,I1)) .+ kron(Dxx3,kron(I2,I1));
end
return L;
end


function speye(n)
	return sparse(1.0I,n,n);
end

function ddxCN(n,h)
# D = ddx(n), 1D derivative operator
	I,J,V = SparseArrays.spdiagm_internal(0 => fill(-(1/h),n), 1 => fill((1/h),n)) 
	return sparse(I, J, V, n, n+1)
end

function av3term(n::Int64,alpha=5/6)
	t = (1-alpha)/2;
	T = spdiagm(0=>alpha*ones(n), 1=>t*ones(n-1),-1=>t*ones(n-1));
	T[1,1] = 1/2 + alpha/2;
	T[end,end] = 1/2 + alpha/2;
	return T;
end


function getNodalSpreadGradients(Msh,avFunc)
	n = Msh.n;
	h = Msh.h;

    if length(n) == 2

        tmp = ddxCN(n[1],h[1]);
        D1 = kron(speye(n[2]+1),tmp);
        D1s = kron(avFunc(n[2]+1),tmp);

        tmp = ddxCN(n[2],h[2])
        D2 = kron(tmp,speye(n[1]+1))
        D2s = kron(tmp,avFunc(n[1]+1))

        G = [D1;D2];
        Gs = [D1s;D2s];

    elseif length(n) == 3

		tmp = ddxCN(n[1],h[1]);
        D1 = kron(speye(n[3]+1),kron(speye(n[2]+1),tmp));
        D1s = 0.5*(kron(speye(n[3]+1),kron(avFunc(n[2]+1),tmp)) + kron(avFunc(n[3]+1),kron(speye(n[2]+1),tmp)));

        tmp = ddxCN(n[2],h[2]);
        D2 = kron(speye(n[3]+1),kron(tmp,speye(n[1]+1)));
        D2s = 0.5*(kron(speye(n[3]+1),kron(tmp,avFunc(n[1]+1))) + kron(avFunc(n[3]+1),kron(tmp,speye(n[1]+1))));

        tmp = ddxCN(n[3],h[3]);
        D3 = kron(tmp,kron(speye(n[2]+1),speye(n[1]+1)));
        D3s = 0.5*(kron(tmp,kron(speye(n[2]+1),avFunc(n[1]+1))) + kron(tmp,kron(avFunc(n[2]+1),speye(n[1]+1))));

        G = [D1;D2;D3];
        Gs = [D1s;D2s;D3s];
    end

    return G,Gs
end

function getSpreadNodalLaplacianAndMass(Mesh,beta)
	# for 2D beta is a scalar and works both for the Laplacian and mass
	# for 3D beta is a vector, where beta[1] is for the Laplacian and beta[2] for the mass

    # Grad  = getNodalGradientMatrix(Msh) 
    # Lap   = Grad'*Grad
    n = Mesh.n;
    if length(n) == 2

        avFunc = n -> av3term(n,0.5);
        G,Gs = getNodalSpreadGradients(Mesh,avFunc);
        Gs = (1-beta)*Gs+beta*G;
        Lap = G'*Gs;

        M = 0.5*kron(av3term(n[2]+1,beta),speye(n[1]+1)) + 0.5*kron(speye(n[2]+1),av3term(n[1]+1,beta));

    elseif length(n) == 3

		if beta == 1
			beta = [1;1]
		end

        avFunc = n -> av3term(n,0.5);
        G,Gs = getNodalSpreadGradients(Mesh,avFunc);
        Gs = (1-beta[1])*Gs+beta[1]*G;
        Lap = G'*Gs;

        third = (1.0/3.0);
        M = third*(kron(speye(n[3]+1),kron(av3term(n[2]+1,beta[2]),speye(n[1]+1))) + 
                   kron(speye(n[3]+1),kron(speye(n[2]+1),av3term(n[1]+1,beta[2]))) + 
                   kron(av3term(n[3]+1,beta[2]),kron(speye(n[2]+1),speye(n[1]+1))));

    end
    
    return Lap,M
end


##
## [0  x2  0]
## [x4 x1 x5]
## [0  x3 0]


function Lap2DStencil(x1::Float64,x2::Float64,x3::Float64,x4::Float64,x5::Float64,h1invsq::Float64,h2invsq::Float64)
	return (2*h1invsq + 2*h2invsq)*x1 - h1invsq*(x2 + x3) - h2invsq*(x4 + x5);
end


function multOpNeumann!(M::RegularMesh,x::Array,y::Array,op::Function)
n = M.n;

if M.dim==2
	n1 = n[1]+1;
	n2 = n[2]+1;
	h1invsq = 1.0./(M.h[1]^2);
	h2invsq = 1.0./(M.h[2]^2);
	@inbounds y[1] = op(x[1],x[1],x[2],x[1],x[n1+1],h1invsq,h2invsq);
	@simd for i=2:n1-1
		@inbounds y[i] = op(x[i],x[i-1],x[i+1],x[i],x[i+n1],h1invsq,h2invsq);
	end
	@inbounds y[n1] = op(x[n1],x[n1-1],x[n1],x[n1],x[n1 + n1],h1invsq,h2invsq);
	for j=2:n2-1
		colShift = n1*(j-1);
		i = 1 + colShift;
		@inbounds y[i] = op(x[i],x[i],x[i+1],x[i-n1],x[i+n1],h1invsq,h2invsq);
		@simd for i = (2 + colShift):(n1-1 + colShift) 
			@inbounds y[i] = op(x[i],x[i-1],x[i+1],x[i-n1],x[i+n1],h1invsq,h2invsq);
		end
		i = n1 + colShift;
		@inbounds y[i] = op(x[i],x[i-1],x[i],x[i-n1],x[i+n1],h1invsq,h2invsq);
	end
	colShift = n1*(n2-1);
	i = 1 + colShift;
	@inbounds y[i] = op(x[i],x[i],x[i+1],x[i-n1],x[i],h1invsq,h2invsq);
	@simd for i = (2 + colShift):(n1-1 + colShift)
		@inbounds y[i] = op(x[i],x[i-1],x[i+1],x[i-n1],x[i],h1invsq,h2invsq);
	end
	i = n1 + colShift;
	@inbounds y[i] = op(x[i],x[i-1],x[i],x[i-n1],x[i],h1invsq,h2invsq);
else
end
end

function multOpDirichlet!(M::RegularMesh,x::Array,y::Array,op::Function)
n = M.n;

if M.dim==2
	n1 = n[1]+1;
	n2 = n[2]+1;
	h1invsq = 1.0./(M.h[1]^2);
	h2invsq = 1.0./(M.h[2]^2);
	@inbounds y[1] = op(x[1],0.0,x[2],0.0,x[n1+1],h1invsq,h2invsq);
	@simd for i=2:n1-1
		@inbounds y[i] = op(x[i],x[i-1],x[i+1],0.0,x[i+n1],h1invsq,h2invsq);
	end
	@inbounds y[n1] = op(x[n1],x[n1-1],0.0,0.0,x[n1 + n1],h1invsq,h2invsq);
	for j=2:n2-1
		colShift = n1*(j-1);
		i = 1 + colShift;
		@inbounds y[i] = op(x[i],0.0,x[i+1],x[i-n1],x[i+n1],h1invsq,h2invsq);
		@simd for i = (2 + colShift):(n1-1 + colShift) 
			@inbounds y[i] = op(x[i],x[i-1],x[i+1],x[i-n1],x[i+n1],h1invsq,h2invsq);
		end
		i = n1 + colShift;
		@inbounds y[i] = op(x[i],x[i-1],0.0,x[i-n1],x[i+n1],h1invsq,h2invsq);
	end
	colShift = n1*(n2-1);
	i = 1 + colShift;
	@inbounds y[i] = op(x[i],0.0,x[i+1],x[i-n1],0.0,h1invsq,h2invsq);
	@simd for i = (2 + colShift):(n1-1 + colShift)
		@inbounds y[i] = op(x[i],x[i-1],x[i+1],x[i-n1],0.0,h1invsq,h2invsq);
	end
	i = n1 + colShift;
	@inbounds y[i] = op(x[i],x[i-1],0.0,x[i-n1],0.0,h1invsq,h2invsq);
else
end
end