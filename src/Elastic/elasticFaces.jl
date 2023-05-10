export getUjProjMatrix

export getVectorMassMatrix

function ddxCN(n::Int64,h) # cells X nodes
	# A = 1/h*spdiagm((-ones(n),ones(n)),[0,1],n,n+1)
	A = spdiagm(n,n+1,0=>fill(-1.0/h,n),1=>fill(1.0/h,n))
	return A	
end


function ddxNC(n::Int64,h) # nodes X cells
	# A = 1/h*spdiagm((-ones(n),ones(n)),[-1,0],n+1,n)
	A = spdiagm(n+1,n,-1=>fill(-1.0/h,n),0=>fill(1.0/h,n))
	# A[1,1] = 0; A[end,end] = 0
	return A
end

# averaging for cell to node
function avNC(n::Int64)
	# A = 1/2*spdiagm((ones(n),ones(n)),[-1,0],n+1,n)
	A = spdiagm(n+1,n,-1=>fill(0.5,n),0=>fill(0.5,n))
	A[1,1] = 1
	A[end,end] = 1
	return A
end	   

speye(n) = spdiagm(0=>fill(1.0,n));

function getDivergenceMatrix(n,h)
	if length(n)==3
		D1 = ddxCN(n[1],h[1])
		D2 = ddxCN(n[2],h[2])
		D3 = ddxCN(n[3],h[3])
		DIV = [kron(speye(n[3]),kron(speye(n[2]),D1))  kron(speye(n[3]),kron(D2,speye(n[1]))) kron(D3,kron(speye(n[2]),speye(n[1])))];
	else
		D1 = ddxCN(n[1],h[1])
		D2 = ddxCN(n[2],h[2])
		DIV = [kron(speye(n[2]),D1)  kron(D2,speye(n[1]))];
	end
	return DIV
end
			 	 
function getDifferentialOperators(n,h,optype = 1)
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
			vectorGrad  = 0.5*[blkdiag(D11,D12,D13);blkdiag(D21,D22,D23);blkdiag(D31,D32,D33)]
			vectorGradT = 0.5*blkdiag([D11;D21;D31],[D12;D22;D32],[D13;D23;D33])
			D11 = 0; D12 = 0; D13 = 0; D22 = 0; D21 = 0; D23 = 0; D31 = 0; D32 = 0; D33 = 0;
			vectorGrad  = (vectorGrad  + vectorGradT);
		else
			vectorGrad = blkdiag([D11;D21;D31],[D12;D22;D32],[D13;D23;D33]);
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
			vectorGrad  = [blkdiag(D11,D12);blkdiag(D21,D22)]
			vectorGradT = blkdiag([D11;D21],[D12;D22])
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

function getEdgeAveragingMatrices(n)
	
	if length(n)==3
		# Average in different direction
		Av1 = avNC(n[1])
		Av2 = avNC(n[2])
		Av3 = avNC(n[3])
	
		# average on x edges
		Ae1 = kron(Av3,kron(Av2,speye(n[1])))
		# average on y edges
		Ae2 = kron(Av3,kron(speye(n[2]),Av1))
		# average on z edges
		Ae3 = kron(speye(n[3]),kron(Av2,Av1))
		return Ae1, Ae2, Ae3;
	else
		# Average in different direction
		Av1 = avNC(n[1]);
		Av2 = avNC(n[2]);
	
		# average on x edges
		Ae1 = kron(Av2,Av1)
		# average on y edges
		Ae2 = Ae1 ; #kron(Av2,Av1)
		return Ae1, Ae2;
	end
	
end




function getTensorMassMatrix(mu,n)
 
	if length(n)==3
		(Ae1, Ae2, Ae3) = getEdgeAveragingMatrices(n)
	
		# M11 = spdiagm(mu[:])
		# M12 = spdiagm(Ae3*mu[:])
		# M13 = spdiagm(Ae2*mu[:])
		# M21 = spdiagm(Ae3*mu[:])
		# M22 = spdiagm(mu[:])
		# M23 = spdiagm(Ae1*mu[:])
		# M31 = spdiagm(Ae2*mu[:])
		# M32 = spdiagm(Ae1*mu[:])
		# M33 = spdiagm(mu[:])
		# M = blkdiag(M11,M12,M13,M21,M22,M23,M31,M32,M33)
	
		mu    = vec(mu);
		Ae3mu = Ae3*mu;
		Ae1mu = Ae1*mu;
		Ae2mu = Ae2*mu;
		m = [mu;Ae3mu;Ae2mu;Ae3mu;mu;Ae1mu;Ae2mu;Ae1mu;mu];
		M = spdiagm(m);
		len = [0;length(mu)+length(Ae3mu)+length(Ae2mu);length(Ae3mu)+length(mu)+length(Ae1mu);length(Ae2mu)+length(Ae1mu)+length(mu)];
		len[3] = len[3]+len[2];
		len[4] = len[4] + len[3];
	else
		(Ae1, Ae2) = getEdgeAveragingMatrices(n)
	
		mu    = vec(mu);
		Ae1mu = Ae1*mu;
		Ae2mu = Ae2*mu;
		m = [mu;Ae1mu;Ae2mu;mu];
		M = spdiagm(0=>m);
		len = [0;length(mu)+length(Ae1mu);length(Ae2mu)+length(mu)];
		len[3] = len[3]+len[2];
	end
	return M,len
end


function getFaceAveragingMatrices(n)
	if length(n) == 3
		# Average in different direction
		Av1 = avNC(n[1])
		Av2 = avNC(n[2])
		Av3 = avNC(n[3])
	
		# average on x faces
		Af1 = kron(speye(n[3]),kron(speye(n[2]),Av1))
		# average on y faces
		Af2 = kron(speye(n[3]),kron(Av2,speye(n[1])))
		# average on z faces
		Af3 = kron(Av3,kron(speye(n[2]),speye(n[1])))
		return Af1, Af2, Af3
	else
		# Average in different direction
		Av1 = avNC(n[1])
		Av2 = avNC(n[2])
	
		# average on x faces
		Af1 = kron(speye(n[2]),Av1)
		# average on y faces
		Af2 = kron(Av2,speye(n[1]))
		return Af1, Af2
	end
	
	
end

function getVectorMassMatrix(mu,n)
	if length(n) == 3
		(Af1, Af2, Af3) = getFaceAveragingMatrices(n)
		mu    = vec(mu);
		m = [Af1*mu;Af2*mu;Af3*mu];
	else
		(Af1, Af2) = getFaceAveragingMatrices(n)
		mu    = vec(mu);
		m = [Af1*mu;Af2*mu];
	end
	
	M = spdiagm(0=>m);
	return M
end

function getVectorMassOp(n)
	if length(n)==3
		(Af1, Af2, Af3) = getFaceAveragingMatrices(n);
		return [Af1;Af2;Af3]
	else
		(Af1, Af2) = getFaceAveragingMatrices(n);
		return [Af1;Af2]
	end
end

function getUjProjMatrix(n,j)
	if length(n)==3
		nf1 = prod(n + [1; 0; 0])
		nf2 = prod(n + [0; 1; 0])
		nf3 = prod(n + [0; 0; 1])
		nf  = [0;nf1;nf2+nf1;nf3+nf1+nf2];
		Pj = speye(nf1+nf2+nf3);
		Pj = Pj[:,nf[j]+1:nf[j+1]];
	else
		nf1 = prod(n + [1; 0])
		nf2 = prod(n + [0; 1])
		nf  = [0;nf1;nf2+nf1];
		Pj = speye(nf1+nf2);
		Pj = Pj[:,nf[j]+1:nf[j+1]];
	end
	return Pj;
end











