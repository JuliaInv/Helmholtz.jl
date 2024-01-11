



export getCellCenteredABL, getShiftedElasticHelmholtzParam, GetElasticHelmholtzOperator, GetElasticHelmholtzShiftOP, GetElasticHelmholtzShiftOPSpread


function GetElasticHelmholtzOperator(p::ElasticHelmholtzParam;spread::Bool = false,beta::Float64 = 1.0)
	if !p.MixedFormulation
		return GetElasticHelmholtzOp(p.Mesh, p.mu,p.lambda,p.rho,p.omega,p.gamma);
	else
		if spread == false
			return GetElasticHelmholtzOpReformulated(p.Mesh, p.mu,p.lambda,p.rho,p.omega,p.gamma);
		else
			return GetElasticHelmholtzOpReformulatedSpread(p.Mesh, p.mu,p.lambda,p.rho,p.omega,p.gamma,beta);
		end	
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

function GetElasticHelmholtzShiftOPSpread(p::ElasticHelmholtzParam,beta, shift::Float64)
	w = p.omega;
	rho = p.rho;
	Mesh = p.Mesh;
	massCells			= -rho[:].*(1im*shift);
	Shift    			    = getFaceMassMatrix(Mesh, massCells.*(1.0./prod(Mesh.h)), saveMat=false, avN2C = avN2C_Nearest);
	Shift  				= -w^2*getSpreadFaceMassMatrix(Mesh.n,beta)*Shift;
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


function GetElasticHelmholtzOp(Mesh::RegularMesh, mu::ArrayTypes,lambda::ArrayTypes,rho::ArrayTypes, omega::Float64, gamma::Array{Float64})
	
	#######################################################################
	# ## OPTION 1 ###########################################################
	# println("Option 1");
	# vecGRAD,vecDiv,Div,nf, = getDifferentialOperators(Mesh,1);
	# Mu     = getTensorMassMatrix(Mesh,mu[:])[1];
	# Lambda = getTensorMassMatrix(Mesh,lambda[:])[1];
	# massCells = rho[:].*((1.0.-(1im/omega)*gamma[:])*(1.0./prod(Mesh.h)));
	# Rho    	   = getFaceMassMatrix(Mesh, massCells, saveMat=false, avN2C = avN2C_Nearest);
	# H1 = vecGRAD'*(Lambda*vecDiv + 2.0*Mu*vecGRAD) - omega^2*Rho;
	# println("nnz: ",length(H.nzval));

	
	
	# ## OPTION 2 ###########################################################
	# println("Option 2");
	# vecGRAD,~,Div,nf, = getDifferentialOperators(Mesh,1);
	# Mu     = getTensorMassMatrix(Mesh,mu[:])[1];
	# massCells = rho[:].*((1.0.-(1im/omega)*gamma[:])*(1.0./prod(Mesh.h)));
	# Rho    	   = getFaceMassMatrix(Mesh, massCells, saveMat=false, avN2C = avN2C_Nearest);
	# LambdaCells = spdiagm(lambda[:]);
	# H2 = Div'*LambdaCells*Div + 2.0*vecGRAD'*(Mu*vecGRAD) - omega^2*Rho;
	
	# println("nnz: ",length(H2.nzval));
	# println(norm(H2-H1,1)/norm(H2,1))
	
	
	# ## OPTION 3 ###########################################################
	# println("Option 3");
	# vecGRAD,vecDiv,Div,nf,vecGRADT = getDifferentialOperators(Mesh,1);
	# Mu     = getTensorMassMatrix(Mesh,mu[:])[1];
	# Lambda = getTensorMassMatrix(Mesh,lambda[:])[1];
	# massCells = rho[:].*((1.0.-(1im/omega)*gamma[:])*(1.0./prod(Mesh.h)));
	# Rho    	   = getFaceMassMatrix(Mesh, massCells, saveMat=false, avN2C = avN2C_Nearest);
	# H3 = vecGRADT'*(Lambda*vecDiv + 2.0*Mu*vecGRAD) - omega^2*Rho;
	# println("nnz: ",length(H3.nzval));
	# println(norm(H3-H2,1)/norm(H2,1));
	
	# ## OPTION 4 ###########################################################
	# println("Option 4");
	vecGRAD,~,Div,nf, = getDifferentialOperators(Mesh,2);
	Mu     = getTensorMassMatrix(Mesh,mu[:])[1];
	LambdaMuCells = spdiagm(0=>(lambda[:] + mu[:]));
	massCells = rho[:].*((1.0.-(1im/omega)*gamma[:])*(1.0./prod(Mesh.h)));
	Rho    	   = getFaceMassMatrix(Mesh, massCells, saveMat=false, avN2C = avN2C_Nearest);
	H = Div'*LambdaMuCells*Div + vecGRAD'*Mu*vecGRAD - omega^2*Rho;
	# println("nnz: ",length(H.nzval));
	# println(norm(H-H2,1)/norm(H,1))
	return H;						
end


function GetElasticHelmholtzOpReformulated(Mesh::RegularMesh,mu::ArrayTypes,lambda::ArrayTypes,rho::ArrayTypes, omega::Float64, gamma::ArrayTypes)
	
	factor = 1.0/(sum(Mesh.h)/length(Mesh.h));
	factor = 1.0;
	vecGRAD,~,Div,nf, = getDifferentialOperators(Mesh,2);
	massCells = rho[:].*((1.0.-(1im/omega)*gamma[:])*(1.0./prod(Mesh.h))); ## we divide by the volume to be compatible with jInv's code.
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

function GetElasticHelmholtzOpReformulatedSpread(Mesh::RegularMesh,mu::Array,lambda::Array,rho::Array, omega::Float64, gamma::Array, beta::Float64 = 5/6)
	factor = 1.0/(sum(Mesh.h)/length(Mesh.h));
	factor = 1.0;
	vectorGrad,Div,nf,vectorGrad_spread,Div_spread = GetDifferentialOperatorsSpreadOpType2(Mesh,beta);
	massCells			= rho[:].*((1.0.-(1im/omega)*gamma[:]));
	Rho    			    = getFaceMassMatrix(Mesh, massCells.*(1.0./prod(Mesh.h)), saveMat=false, avN2C = avN2C_Nearest);
	Rho  				= getSpreadFaceMassMatrix(Mesh.n,beta)*Rho;
	Mu     			    = getTensorMassMatrix(Mesh,mu[:])[1];
	A 				    = vectorGrad'*Mu*vectorGrad_spread - omega^2*Rho;
	vectorGrad = vectorGrad_spread = Mu = Rho = []
	C 					= spdiagm(0=>(-factor^2)./(lambda[:]+mu[:]));
	Div.nzval         .*= (-factor);
	Div_spread.nzval  .*= (-factor);
	Div = convert(SparseMatrixCSC{eltype(mu),Int64},Div)
	Div_spread = convert(SparseMatrixCSC{eltype(mu),Int64},Div_spread)
	C = convert(SparseMatrixCSC{eltype(mu),Int64},C)
	A = convert(SparseMatrixCSC{complex(eltype(mu)),Int64},A)
	H = [A  Div' ; Div_spread  C;];
	return H;						
end



