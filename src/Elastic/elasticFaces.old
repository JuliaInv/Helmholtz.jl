function ddx(n)
# D = ddx(n), 1D derivative operator
	I,J,V = SparseArrays.spdiagm_internal(0 => fill(-1.0,n), 1 => fill(1.0,n)) 
	return sparse(I, J, V, n, n+1)
end

function ddxCN(n::Int64,h) ## From nodes to cells
	D = (1/h)*ddx(n);
	return D
end

function ddxNC(n::Int64,h) ## From cells to nodes
	#A = 1/h*spdiagm((-ones(n),ones(n)),[-1,0],n+1,n)
	I, J, V = SparseArrays.spdiagm_internal(-1 => fill(-1.0,n), 0 => fill(1.0,n)); 
	D = sparse(I, J, (1/h)*V,n+1,n);
	
	# D = spdiagm(n+1,n,-1=>fill(-1.0/h,n),0=>fill(1.0/h,n))
	## This is for Neumann boundary conditions: need to figure this out
	#D[1,1] = 0.0;
	#D[end,end] = 0.0;
	return D
end

 	 
function getDifferentialOperators(M::RegularMesh,optype = 1)
	n = M.n;
	h = M.h;
	if length(n)==3
		# Face sizes
		nf1 = prod(n + [1; 0; 0])
		nf2 = prod(n + [0; 1; 0])
		nf3 = prod(n + [0; 0; 1])
		nf  = [nf1; nf2; nf3];
		
		# Notation Dij = derivative of component j in direction i
		tmp = ddxCN(n[1],h[1])
		D11 = kron(speye(n[3]),kron(speye(n[2]),tmp))

		tmp = ddxNC(n[1],h[1])
		D12 = kron(speye(n[3]),kron(speye(n[2]+1),tmp))

		tmp = ddxNC(n[1],h[1])
		D13 = kron(speye(n[3]+1),kron(speye(n[2]),tmp))

		tmp = ddxNC(n[2],h[2])
		D21 = kron(speye(n[3]),kron(tmp,speye(n[1]+1)))

		tmp = ddxCN(n[2],h[2])
		D22 = kron(speye(n[3]),kron(tmp,speye(n[1])))

		tmp = ddxNC(n[2],h[2])
		D23 = kron(speye(n[3]+1),kron(tmp,speye(n[1])))

		tmp = ddxNC(n[3],h[3])
		D31 = kron(tmp,kron(speye(n[2]),speye(n[1]+1)))

		tmp = ddxNC(n[3],h[3])
		D32 = kron(tmp,kron(speye(n[2]+1),speye(n[1])))

		tmp = ddxCN(n[3],h[3])
		D33 = kron(tmp,kron(speye(n[2]),speye(n[1])))

		# Tensor sizes
		t = [size(D11,1); size(D12,1); size(D13,1);
			size(D21,1); size(D22,1); size(D23,1);
			size(D31,1); size(D32,1); size(D33,1);]
		
		
		Div = [D11 D22 D33]
		if optype==1
			vectorDiv = [Div; spzeros(t[2]+t[3]+t[4],sum(nf)); Div; spzeros(t[6]+t[7]+t[8],sum(nf)); Div];
			vectorGrad  = 0.5*[blockdiag(D11,D12,D13);blockdiag(D21,D22,D23);blockdiag(D31,D32,D33)]
			vectorGradT = 0.5*blockdiag([D11;D21;D31],[D12;D22;D32],[D13;D23;D33])
			D11 = 0; D12 = 0; D13 = 0; D22 = 0; D21 = 0; D23 = 0; D31 = 0; D32 = 0; D33 = 0;
			vectorGrad  = (vectorGrad  + vectorGradT);
		else
			vectorGrad = blockdiag([D11;D21;D31],[D12;D22;D32],[D13;D23;D33]);
			vectorDiv = 0;
			vectorGradT = 0;
		end
	else		
		nf1 = prod(n + [1; 0]);
		nf2 = prod(n + [0; 1]);
		nf  = [nf1; nf2];
		
		# Notation Dij = derivative of component j in direction i
		tmp = ddxCN(n[1],h[1]);
		D11 = kron(speye(n[2]),tmp);

		tmp = ddxNC(n[1],h[1]);
		D12 = kron(speye(n[2]+1),tmp);

		tmp = ddxNC(n[2],h[2])
		D21 = kron(tmp,speye(n[1]+1))

		tmp = ddxCN(n[2],h[2])
		D22 = kron(tmp,speye(n[1]))

		# Tensor sizes
		t = [size(D11,1); size(D12,1); size(D21,1); size(D22,1);]
		  
		Div = [D11 D22]
		
		
		if optype==1
			vectorDiv = [Div; spzeros(t[2]+t[3],sum(nf)); Div; ];
			vectorGrad  = [blockdiag(D11,D12);blockdiag(D21,D22)]
			vectorGradT = blockdiag([D11;D21],[D12;D22])
			vectorGrad  = 0.5*(vectorGrad  + vectorGradT)
		else
			# vectorGrad = [[D11;D21],[D12;D22]];
			vectorGrad = blockdiag([D11;D21],[D12;D22]); ## diagGrad
			vectorDiv = 0;
			vectorGradT = 0;
		end
	end 
	return vectorGrad, vectorDiv, Div,nf,vectorGradT
end

