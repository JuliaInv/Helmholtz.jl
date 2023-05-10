



export getCellCenteredABL, getShiftedElasticHelmholtzParam, GetElasticHelmholtzOperator, GetElasticHelmholtzShiftOP


function GetElasticHelmholtzOperator(p::ElasticHelmholtzParam)
	if !p.MixedFormulation
		return GetElasticHelmholtzOp(p.Mesh, p.mu,p.lambda,p.rho,p.omega,p.gamma);
	else
		return GetElasticHelmholtzOpReformulated(p.Mesh, p.mu,p.lambda,p.rho,p.omega,p.gamma);
	end
end


function getShiftedElasticHelmholtzParam(p::ElasticHelmholtzParam,s::Float64)
	return ElasticHelmholtzParam(p.Mesh,p.omega,p.lambda,p.rho,p.mu,p.gamma + s*p.omega,p.NeumannOnTop,p.MixedFormulation);
end

function GetElasticHelmholtzShiftOP(p::ElasticHelmholtzParam,shift::Float64)
	w = p.omega;
	rho = p.rho;
	Mesh = p.Mesh;
	m = (1im*shift)*(w^2)*(rho[:].*(1.0./prod(Mesh.h)));
	Shift = getFaceMassMatrix(Mesh,m, saveMat=false, avN2C = avN2C_Nearest);
	if p.MixedFormulation
		Shift = blockdiag(Shift,spzeros(prod(p.Mesh.n),prod(p.Mesh.n)));
	end
	return Shift;

end




function getCellCenteredABL(Mesh::RegularMesh,NeumannAtFirstDim::Bool,ABLpad::Array{Int64},ABLamp::Float64)	
return getABL(Mesh.n,NeumannAtFirstDim,ABLpad,ABLamp);			
end

function speye(n)
	return sparse(1.0I,n,n);
end

#function spdiagm(x::Vector)
#	return sparse(Diagonal(x));
#end


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






function GetElasticHelmholtzOp(Mesh::RegularMesh, mu::ArrayTypes,lambda::ArrayTypes,rho::ArrayTypes, omega::Float64, gamma::Array{Float64})
	############### JUST FOR TIMING - IGNORE ###########################
	# vecGRAD,vecDiv,Div,nf, = getDifferentialOperators(Mesh.n,Mesh.h);
	# Mu     = getTensorMassMatrix(mu[:],Mesh.n)[1];
	# Lambda = getTensorMassMatrix(lambda[:],Mesh.n)[1];
	# Rho	   = getVectorMassMatrix(rho[:].*(1-(1im/omega)*gamma[:]),Mesh.n);
	# H = vecGRAD'*(Lambda*vecDiv + 2.0*Mu*vecGRAD) - omega^2*Rho;
	#######################################################################
	# ## OPTION 1 ###########################################################
	# println("Option 1");
	# tic()
	# vecGRAD,vecDiv,Div,nf, = getDifferentialOperators(Mesh.n,Mesh.h);
	# Mu     = getTensorMassMatrix(mu[:],Mesh.n)[1];
	# Lambda = getTensorMassMatrix(lambda[:],Mesh.n)[1];
	# Rho	   = getVectorMassMatrix(rho[:].*(1-(1im/omega)*gamma[:]),Mesh.n);
	# H = vecGRAD'*(Lambda*vecDiv + 2.0*Mu*vecGRAD) - omega^2*Rho;
	# toc()
	# println("nnz: ",length(H.nzval));

	
	
	# ## OPTION 2 ###########################################################
	# println("Option 2");
	# vecGRAD,~,Div,nf, = getDifferentialOperators(Mesh.n,Mesh.h);
	# Mu     = getTensorMassMatrix(mu[:],Mesh.n)[1];
	# Rho	   = getVectorMassMatrix(rho[:].*(1-(1im/omega)*gamma[:]),Mesh.n);	
	# LambdaCells = spdiagm(lambda[:]);
	# H2 = Div'*LambdaCells*Div + 2.0*vecGRAD'*(Mu*vecGRAD) - omega^2*Rho;
	# println("nnz: ",length(H2.nzval));
	# println(norm(H2-H,1)/norm(H,1))
	
	
	# ## OPTION 3 ###########################################################
	# println("Option 3");
	# vecGRAD,vecDiv,Div,nf,vecGRADT = getDifferentialOperators(Mesh.n,Mesh.h);
	# Mu     = getTensorMassMatrix(mu[:],Mesh.n)[1];
	# Lambda = getTensorMassMatrix(lambda[:],Mesh.n)[1];
	# Rho	   = getVectorMassMatrix(rho[:].*(1-(1im/omega)*gamma[:]),Mesh.n);
	# H3 = vecGRADT'*(Lambda*vecDiv + 2.0*Mu*vecGRAD) - omega^2*Rho;
	# println("nnz: ",length(H3.nzval));
	# println(norm(H3-H,1)/norm(H,1));
	
	# ## OPTION 4 ###########################################################
	# println("Option 4");
	vecGRAD,~,Div,nf, = getDifferentialOperators(Mesh,2);
	Mu     = getTensorMassMatrix(Mesh,mu[:])[1];
	LambdaMuCells = spdiagm(0=>(lambda[:] + mu[:]));
	massCells = rho[:].*((1.0.-(1im/omega)*gamma[:])*(1.0./prod(Mesh.h)));
	# Rho	   = getVectorMassMatrix(massCells,Mesh.n);
	Rho    	   = getFaceMassMatrix(Mesh, massCells, saveMat=false, avN2C = avN2C_Nearest);
	H = Div'*LambdaMuCells*Div + vecGRAD'*Mu*vecGRAD - omega^2*Rho;
	# println("nnz: ",length(H4.nzval));
	# println(norm(H4-H,1)/norm(H,1))
	return H;						
end


function GetElasticHelmholtzOpReformulated(Mesh::RegularMesh,mu::Array{Float64},lambda::Array{Float64},rho::Array{Float64}, omega::Float64, gamma::Array{Float64})
	
	factor = 1.0/(sum(Mesh.h)/length(Mesh.h));
	factor = 1.0;
	vecGRAD,~,Div,nf, = getDifferentialOperators(Mesh,2);
	massCells = rho[:].*((1.0.-(1im/omega)*gamma[:])*(1.0./prod(Mesh.h)));
	# Rho	    		  = getVectorMassMatrix(massCells,Mesh.n);
	Rho    			  = getFaceMassMatrix(Mesh, massCells, saveMat=false, avN2C = avN2C_Nearest);
	Mu     			  = getTensorMassMatrix(Mesh,mu[:])[1];
	A 				  = vecGRAD'*Mu*vecGRAD - omega^2*Rho;
	C = spdiagm(0=>(-factor^2)./(lambda[:]+mu[:]));
	Div.nzval .*= (-factor);
	H = [A  Div' ; Div  C;];
	
	#p = -(lam+mu)*div(u)/h
	
	
	# (vecGRAD,~,Div,nf) = getDifferentialOperators(Mesh,1);
	# Mu,len      = getTensorMassMatrix(mu[:],Mesh.n);
	# MuCells = spdiagm(mu[:]);
	# Rho	   = getVectorMassMatrix(rho[:].*(1-(1im/omega)*gamma[:]),Mesh.n);
	# C = (spdiagm(-(factor^2)./(lambda[:]+mu[:])));
	# H1 = [(2.0*vecGRAD'*(Mu*vecGRAD) - Div'*MuCells*Div  - omega^2*Rho)  -factor*Div' ; -factor*Div  C;];
	
	return H;						
end


