function getElasticHelmholtzProblem(;n,w_factor=0.9,model="linear",rho = 1.0, lambda = 1.0, mu = 1.0,pad=8,TYPE = ComplexF64,reformulated = false,shift=0.25)

if model == "const"
	(rho,lambda,mu,Minv) = getModel("const", n, lambda, mu, rho);
else
	(rho,lambda,mu,Minv) = getModel(model, n);
end

q = getElasticPointSource(Minv,TYPE);
sp = sqrt.(rho[:]./(lambda[:] + 2*mu[:]))
ss = sqrt.(rho[:]./(mu[:]))
w = w_factor*getMaximalFrequency(ss.^2,Minv);

print("sp*omega*h = ",w*maximum(Minv.h)*maximum(sp),";    ");
println("ss*omega*h = ",w*maximum(Minv.h)*maximum(ss));
pad = pad*ones(Int64,3);
gamma = getCellCenteredABL(Minv,true,pad,1.0/(sum(Minv.h)/length(Minv.h)));
#gamma .+= 0.01*pi;
gamma .+= 0.001*pi;
#gamma .+= 0.005*w;

Hparam 		= ElasticHelmholtzParam(Minv,w,lambda,rho,mu,gamma,true,reformulated)
Shift 		= GetElasticHelmholtzShiftOP(Hparam,shift)';
Shift       = convert(SparseMatrixCSC{TYPE,spIndType},Shift);

HrT = sparse(GetElasticHelmholtzOperator(Hparam)');
if Hparam.MixedFormulation
	q = [q;zeros(eltype(q),prod(Minv.n))];
end
return Minv,Hparam,q,HrT,Shift,w
end

@everywhere begin
function getMassOmega(DDparam,i,w)
#	meanRho 	= sum(Hparam.rho[:])./length(Hparam.rho[:]);
	meanRho 	= 1.0;
	NumCells 	= DDparam.numDomains;
	overlap 	= DDparam.overlap;
	M 			= DDparam.Mesh;
	subMesh   	= getSubMeshOfCell(NumCells,overlap,i,M);
	IIp       	= getCellCenteredIndicesOfCell(NumCells,overlap,i,M.n);
	code 	  	= [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2]];
	if length(i) == 3
		code 	  	= [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2] ; i[3]!=1 i[3]!=NumCells[3] ];
	end
	m  			= getABL(subMesh.n,i[end]==1,overlap,1.0/(sum(M.h)/length(M.h)),code);
	m			= convert(Array{ComplexF64,1},m[:]);
	m 			.*= meanRho*(1im*(w^2)*(1.0./prod(subMesh.h)));
	MassT 		= getFaceMassMatrix(subMesh,m, saveMat=false, avN2C = avN2C_Nearest);
	MassT 		= blockdiag(MassT,spzeros(prod(subMesh.n),prod(subMesh.n)));
	MassT 		= convert(SparseMatrixCSC{TYPE,ITYPE},MassT);
	return MassT;
end


function getSubParamsFunc(Hparam, M::RegularMesh,i::Array{Int64},NumCells::Array{Int64},overlap::Array{Int64},shift)
	subMesh   = getSubMeshOfCell(NumCells,overlap,i,M);
	IIp       = getCellCenteredIndicesOfCell(NumCells,overlap,i,M.n);
	code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2]];
	if length(i) == 3
		code 	  	= [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2] ; i[3]!=1 i[3]!=NumCells[3] ];
	end
	# for 2D results it was .+1 below.
	# subgamma  = getABL(subMesh.n,i[end]==1,overlap.+1,2.0./(sum(M.h)/length(M.h)),code);
	subgamma  = getABL(subMesh.n,i[end]==1,overlap.+1,2.0./(sum(M.h)/length(M.h)),code);
	subgamma  = subgamma[:].*sqrt.(Hparam.mu[IIp]./Hparam.rho[IIp]) .+ shift*Hparam.omega;
	#subgamma  .+= shift*Hparam.omega;
	# figure();imshow(reshape(sqrt.(Hparam.mu[IIp]./Hparam.rho[IIp]),tuple(subMesh.n...))');colorbar(); title(string("code = ",code," i = ",i));
	# figure();imshow(reshape(subgamma,tuple(subMesh.n...))'); title(string("code = ",code," i = ",i));
	Hparam    = ElasticHelmholtzParam(subMesh,Hparam.omega,Hparam.lambda[IIp],Hparam.rho[IIp],Hparam.mu[IIp],Hparam.gamma[IIp]+subgamma[:],false,Hparam.MixedFormulation);
	return Hparam;
end
end



