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
	return D
end

function av3term(n::Int64,alpha::Float64)
	t = (1-alpha)/2;
	T = spdiagm(0=>alpha*ones(n), 1=>t*ones(n-1),-1=>t*ones(n-1));
	T[1,1] = 1/2 + alpha/2;
	T[end,end] = 1/2 + alpha/2;
	return T;
end

function av3termGrad(n::Int64)
	t = 0.25;
	T = spdiagm(0=>0.5*ones(n), 1=>0.25*ones(n-1),-1=>0.25*ones(n-1));
	T[1,1] = 0.75;
	T[end,end] = 0.75;
	return T;
end


function getGradOps(n,h,avFunc = speye,beta = 1.0)
# Notation Dij = derivative of component j in direction i
	if length(n)==2
		tmp = ddxCN(n[1],h[1]);
		D11 = kron(speye(n[2]),tmp);
		D11s = kron(avFunc(n[2]),tmp);
		D11s = beta*D11 + (1-beta)*D11s; 
		
		tmp  = ddxNC(n[1],h[1]);
		D12  = kron(speye(n[2]+1),tmp);
		D12s = kron(avFunc(n[2]+1),tmp);
		D12s = beta*D12 + (1-beta)*D12s;

		tmp = ddxNC(n[2],h[2])
		D21 = kron(tmp,speye(n[1]+1))
		D21s = kron(tmp,avFunc(n[1]+1))
		D21s = beta*D21 + (1-beta)*D21s;

		tmp = ddxCN(n[2],h[2])
		D22 = kron(tmp,speye(n[1]))
		D22s = kron(tmp,avFunc(n[1]))
		D22s = beta*D22 + (1-beta)*D22s;
		return D11,D12,D21,D22,D11s,D12s,D21s,D22s
	elseif length(n)==3
		tmp = ddxCN(n[1],h[1])
		D11 = kron(speye(n[3]),kron(speye(n[2]),tmp))
		D11s = kron(avFunc(n[3]),kron(avFunc(n[2]),tmp))
		# D11s = 0.5*(kron(speye(n[3]),kron(avFunc(n[2]),tmp)) + kron(avFunc(n[3]),kron(speye(n[2]),tmp)))
		D11 = convert(SparseMatrixCSC{Float32,Int64},D11)
		D11s = convert(SparseMatrixCSC{Float32,Int64},beta*D11 + (1-beta)*D11s)

		tmp = ddxNC(n[1],h[1])
		D12 = kron(speye(n[3]),kron(speye(n[2]+1),tmp))
		D12s = kron(avFunc(n[3]),kron(avFunc(n[2]+1),tmp))
		# D12s = 0.5*(kron(speye(n[3]),kron(avFunc(n[2]+1),tmp)) + kron(avFunc(n[3]),kron(speye(n[2]+1),tmp)))
		D12 = convert(SparseMatrixCSC{Float32,Int64},D12)
		D12s = convert(SparseMatrixCSC{Float32,Int64},beta*D12 + (1-beta)*D12s)
		
		
		tmp = ddxNC(n[1],h[1])
		D13 = kron(speye(n[3]+1),kron(speye(n[2]),tmp))
		D13s = kron(avFunc(n[3]+1),kron(avFunc(n[2]),tmp))
		# D13s = 0.5*(kron(speye(n[3]+1),kron(avFunc(n[2]),tmp)) + kron(avFunc(n[3]+1),kron(speye(n[2]),tmp)))
		D13 = convert(SparseMatrixCSC{Float32,Int64},D13)
		D13s = convert(SparseMatrixCSC{Float32,Int64},beta*D13 + (1-beta)*D13s)
		
		
		tmp = ddxNC(n[2],h[2])
		D21 = kron(speye(n[3]),kron(tmp,speye(n[1]+1)))
		D21s = kron(avFunc(n[3]),kron(tmp,avFunc(n[1]+1)))
		# D21s = 0.5*(kron(speye(n[3]),kron(tmp,avFunc(n[1]+1))) + kron(avFunc(n[3]),kron(tmp,speye(n[1]+1))))
		D21 = convert(SparseMatrixCSC{Float32,Int64},D21)
		D21s = convert(SparseMatrixCSC{Float32,Int64},beta*D21 + (1-beta)*D21s)
		
		
		tmp = ddxCN(n[2],h[2])
		D22 = kron(speye(n[3]),kron(tmp,speye(n[1])))
		D22s = kron(avFunc(n[3]),kron(tmp,avFunc(n[1])))
		# D22s = 0.5*(kron(speye(n[3]),kron(tmp,avFunc(n[1]))) + kron(avFunc(n[3]),kron(tmp,speye(n[1]))))
		D22 = convert(SparseMatrixCSC{Float32,Int64},D22)
		D22s = convert(SparseMatrixCSC{Float32,Int64},beta*D22 + (1-beta)*D22s)
		
		
		tmp = ddxNC(n[2],h[2])
		D23 = kron(speye(n[3]+1),kron(tmp,speye(n[1])))
		D23s = kron(avFunc(n[3]+1),kron(tmp,avFunc(n[1])))
		# D23s = 0.5*(kron(speye(n[3]+1),kron(tmp,avFunc(n[1]))) + kron(avFunc(n[3]+1),kron(tmp,speye(n[1]))))
		D23 = convert(SparseMatrixCSC{Float32,Int64},D23)
		D23s = convert(SparseMatrixCSC{Float32,Int64},beta*D23 + (1-beta)*D23s)
		
		
		
		tmp = ddxNC(n[3],h[3])
		D31 = kron(tmp,kron(speye(n[2]),speye(n[1]+1)))
		D31 = convert(SparseMatrixCSC{Float32,Int64},D31)
		D31s= kron(tmp,kron(avFunc(n[2]),avFunc(n[1]+1)))
		# D31s= 0.5*(kron(tmp,kron(speye(n[2]),avFunc(n[1]+1))) + kron(tmp,kron(avFunc(n[2]),speye(n[1]+1))))
		D31s = convert(SparseMatrixCSC{Float32,Int64},beta*D31 + (1-beta)*D31s)
		

		tmp = ddxNC(n[3],h[3])
		D32 = kron(tmp,kron(speye(n[2]+1),speye(n[1])))
		D32 = convert(SparseMatrixCSC{Float32,Int64},D32)
		D32s = kron(tmp,kron(avFunc(n[2]+1),avFunc(n[1])))
		# D32s = 0.5*(kron(tmp,kron(speye(n[2]+1),avFunc(n[1]))) + kron(tmp,kron(avFunc(n[2]+1),speye(n[1]))))
		D32s = convert(SparseMatrixCSC{Float32,Int64},beta*D32s + (1-beta)*D32s)

		tmp = ddxCN(n[3],h[3])
		D33 = kron(tmp,kron(speye(n[2]),speye(n[1])))
		D33 = convert(SparseMatrixCSC{Float32,Int64},D33)
		D33s = kron(tmp,kron(avFunc(n[2]),avFunc(n[1])))
		# D33s = 0.5*(kron(tmp,kron(speye(n[2]),avFunc(n[1]))) + kron(tmp,kron(avFunc(n[2]),speye(n[1]))))
		D33s = convert(SparseMatrixCSC{Float32,Int64},beta*D33 + (1-beta)*D33s)

		return D11, D12, D13, D21, D22, D23, D31, D32, D33, D11s, D12s, D13s, D21s, D22s, D23s, D31s, D32s, D33s
		# t = [size(D11,1); size(D12,1); size(D13,1);
			# size(D21,1); size(D22,1); size(D23,1);
			# size(D31,1); size(D32,1); size(D33,1);]
		
		
		# Div = [D11 D22 D33]
		# if optype==1
			# vectorDiv = [Div; spzeros(t[2]+t[3]+t[4],sum(nf)); Div; spzeros(t[6]+t[7]+t[8],sum(nf)); Div];
			# vectorGrad  = 0.5*[blockdiag(D11,D12,D13);blockdiag(D21,D22,D23);blockdiag(D31,D32,D33)]
			# vectorGradT = 0.5*blockdiag([D11;D21;D31],[D12;D22;D32],[D13;D23;D33])
			# D11 = 0; D12 = 0; D13 = 0; D22 = 0; D21 = 0; D23 = 0; D31 = 0; D32 = 0; D33 = 0;
			# vectorGrad  = (vectorGrad  + vectorGradT);
		# else
			# vectorGrad = blockdiag([D11;D21;D31],[D12;D22;D32],[D13;D23;D33]);
			# vectorDiv = 0;
			# vectorGradT = 0;
		# end
	end
end



function GetDifferentialOperatorsSpreadOpType2(M::RegularMesh,beta::Float64)
	n = M.n;
	h = M.h;
	if length(n)==2
		nf1 = prod(n + [1; 0]);
		nf2 = prod(n + [0; 1]);
		nf  = [nf1; nf2];
		# println("Placing 2/3")
		D11,D12,D21,D22,D11s,D12s,D21s,D22s = getGradOps(n,h,av3termGrad,beta);
		Div = [D11 D22]
		vectorGrad = blockdiag([D11;D21],[D12;D22]); ## diagGrad
		D11 = D12 = D21 = D22 = [];
		Div_spread        = [D11s D22s]
		vectorGrad_spread = blockdiag([D11s;D21s],[D12s;D22s]); ## diagGrad
		return vectorGrad,Div,nf,vectorGrad_spread,Div_spread
	elseif length(n)==3
		# Face sizes
		nf1 = prod(n + [1; 0; 0])
		nf2 = prod(n + [0; 1; 0])
		nf3 = prod(n + [0; 0; 1])
		nf  = [nf1; nf2; nf3];
		D11, D12, D13, D21, D22, D23, D31, D32, D33, D11s, D12s, D13s, D21s, D22s, D23s, D31s, D32s, D33s = getGradOps(n,h,av3termGrad,beta);
		Div = [D11 D22 D33];
		vectorGrad = convert(SparseMatrixCSC{Float32,Int64},blockdiag([D11;D21;D31],[D12;D22;D32],[D13;D23;D33]));
		D11 = D12 = D13 = D21 = D22 = D23 = D31 = D32 = D33 = [];
		Div_spread        = convert(SparseMatrixCSC{Float32,Int64},[D11s D22s D33s]);
		vectorGrad_spread = convert(SparseMatrixCSC{Float32,Int64},blockdiag([D11s;D21s;D31s],[D12s;D22s;D32s],[D13s;D23s;D33s]));
		return vectorGrad,Div,nf,vectorGrad_spread,Div_spread
	end	
	return 0; 
end

function getSpreadFaceMassMatrix(n,beta::Float64)
	if length(n)==2
		# A1 = 0.5*kron(av3term(n[2]+1,beta),speye(n[1])) + 0.5*kron(speye(n[2]+1),av3term(n[1],beta));
		# A2 = 0.5*kron(speye(n[2]),av3term(n[1]+1,beta)) + 0.5*kron(av3term(n[2],beta),speye(n[1]+1));
		A1 = 0.5*kron(av3term(n[2],beta),speye(n[1]+1)) + 0.5*kron(speye(n[2]),av3term(n[1]+1,beta));
		A2 = 0.5*kron(speye(n[2]+1),av3term(n[1],beta)) + 0.5*kron(av3term(n[2]+1,beta),speye(n[1]));
		M = blockdiag(A1,A2);
	elseif length(n)==3
		third = (1.0/3.0);
		A1 = third*kron(speye(n[3]),kron(av3term(n[2],beta),speye(n[1]+1))) 
		     + third*kron(speye(n[3]),kron(speye(n[2]),av3term(n[1]+1,beta))) 
			 + third*kron(av3term(n[3],beta),kron(speye(n[2]),speye(n[1]+1)));
		A2 = third*kron(speye(n[3]),kron(speye(n[2]+1),av3term(n[1],beta))) 
		     + third*kron(speye(n[3]),kron(av3term(n[2]+1,beta),speye(n[1]))) 
			 + third*kron(av3term(n[3],beta),kron(speye(n[2]+1),speye(n[1]))); 
		A3 = third*kron(speye(n[3]+1),kron(speye(n[2]),av3term(n[1],beta))) 
		     + third*kron(speye(n[3]+1),kron(av3term(n[2],beta),speye(n[1]))) 
			 + third*kron(av3term(n[3]+1,beta),kron(speye(n[2]),speye(n[1]))); 
		M = blockdiag(A1,A2,A3);
	end
	return M
end


