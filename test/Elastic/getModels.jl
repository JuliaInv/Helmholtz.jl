#using jInvSeismic.Utils

function getModel(model, n, lambda_param = 1.0, mu_param = 1.0, rho_param = 1.0)
if model=="linear"
	domain = [0.0,16.0,0.0,5.0];
	if length(n) == 3
		domain = [0.0,16.0,0.0,16.0,0.0,5.0];
	end

	rho    = getLinearModel(2.0,3.0,n);
	lambda = getLinearModel(4.0,20.0,n);
	mu     = getLinearModel(1.0,15.0,n);
	
	
	# rho    = getLinearModel(1.7,2.7,n);
	# lambda = getLinearModel(4.2,48.6,n);
	# mu     = getLinearModel(1.75,24.0,n);
	
	
	Vs = sqrt.(mu./rho);
	Vp = sqrt.((lambda+2*mu)./rho)
	Poisson = (lambda./(2*(lambda+mu)));
	M = getRegularMesh(domain,n);
	# figure();
	# subplot(2,2,1);
	# plotModel(Vp,includeMeshInfo = true,M_regular = M);
	# title("Pressure Velocity");
	# subplot(2,2,2);
	# plotModel(Vs,includeMeshInfo = true,M_regular = M);
	# title("Shear Velocity");
	# subplot(2,2,3);
	# plotModel(rho,includeMeshInfo = true,M_regular = M);
	# title("Density");
	# subplot(2,2,4);
	# plotModel(Poisson,includeMeshInfo = true,M_regular = M);
	# title("Poisson's ratio");	
	# println("IM HERE!!!")

elseif model=="Marmousi"
	if length(n)==3
		error("Marmousi is only 2D");
	end
	domain = [0.0,17.0,0.0,3.5]; ## that's the original domain
	#model_dir = "./../../../BenchmarkModels/";
	model_dir = "./../BenchmarkModels/";
	# Size of Marmousi: [13601;2801]. domain = 
	# file = matopen(string(model_dir,"ElasticMarmousi2_single.mat")); 
	# DICT = read(file); close(file);
	# Vs = DICT["Vs"]';
	# println(size(Vs))
	# Vp = DICT["Vp"]';
	# rho = DICT["rho"]';
	# Vs = Vs[1:end,370:end];
	# Vp = Vp[1:end,370:end];
	# rho = rho[1:end,370:end];
	
	domain_data = [0.0,17.0,0.0,3.0];
	println("Reading Marmousi")
	Vs = readdlm(string(model_dir,"MarmousiVs_small.dat"))/1000.0
	println("1")
	Vp = readdlm(string(model_dir,"MarmousiVp_small.dat"))/1000.0
	println("2")
	rho = readdlm(string(model_dir,"MarmousiRho_small.dat"))/1000.0
	println("Finished reading Marmousi")
	
	n_data = collect(size(Vs));
	M = getRegularMesh(domain_data,n_data);
	
	Vs = expandModelNearest(smoothModel(Vs,[],0),n_data,n);
	Vp = expandModelNearest(smoothModel(Vp,[],0),n_data,n);
	rho = expandModelNearest(smoothModel(rho,[],0),n_data,n);
	
	# writedlm(string(model_dir,"MarmousiVs_small.dat"),round.(Int16,Vs*1000.0))
	# writedlm(string(model_dir,"MarmousiVp_small.dat"),round.(Int16,Vp*1000))
	# writedlm(string(model_dir,"MarmousiRho_small.dat"),round.(Int16,rho*1000))
	n_data = collect(size(Vs));
	
	
	mu = rho.*Vs.^2
	lambda = rho.*(Vp.^2 - 2*Vs.^2)
	poisson = (lambda./(2*(lambda+mu)));
	
	# figure();
	# subplot(2,2,1);
	# plotModel(Vp,includeMeshInfo = true, M_regular = M);
	# title("Pressure Velocity");
	# subplot(2,2,2);
	# plotModel(Vs,includeMeshInfo = true, M_regular = M);
	# title("Shear Velocity");
	# subplot(2,2,3);
	# plotModel(rho,includeMeshInfo = true, M_regular = M);
	# title("Density");
	# subplot(2,2,4);
	# plotModel(poisson,includeMeshInfo = true, M_regular = M);
	# title("Poisson's ratio");	
	# error("IM HERE")
	# pad_down =  2^round(Int64, (0.5/3.0)*n[2]);
	pad_down = 16;
	n_new = tuple((collect(size(Vs)) + [0;pad_down])...);
	Vs_new = zeros(n_new);Vs_new[:,1:size(Vs,2)] = Vs;
	Vp_new = zeros(n_new);Vp_new[:,1:size(Vs,2)] = Vp;
	rho_new = zeros(n_new);rho_new[:,1:size(Vs,2)] = rho;
	for k=0:pad_down-1
		Vs_new[:,end-k] = Vs[:,end];
		Vp_new[:,end-k] = Vp[:,end];
		rho_new[:,end-k] = rho[:,end];
	end
	Vs = Vs_new;
	Vp = Vp_new;
	rho = rho_new;
	mu = rho.*Vs.^2;
	lambda = rho.*(Vp.^2 - 2*Vs.^2);
	domain_data = [0.0,17.0,0.0,3.0 + pad_down*M.h[2]];
	n_data = collect(size(rho));
	M = getRegularMesh(domain_data,n_data);
	# println(n_data)
	# poisson = (lambda./(2*(lambda+mu)));	
elseif model=="SEAM"


elseif model=="const"
	n_tup = tuple(n...);
	rho    = rho_param*ones(n_tup);
	lambda = lambda_param*ones(n_tup);
	mu     = mu_param*ones(n_tup);
	domain = [0.0,16.0,0.0,5.0];
	if length(n) == 3
		domain = [0.0,16.0,0.0,16.0,0.0,5.0];
	end
	M = getRegularMesh(domain,n);
elseif model=="Overthrust"
	domain = [0.0,20.0,0.0,20.0,0.0,4.65];
	model_dir = "./../../../BenchmarkModels/";
	model_dir = "./../BenchmarkModels/";
	# file = matopen(string(model_dir,"3DOverthrust801801187.mat")); 
	# DICT = read(file); close(file);
	# m = DICT["A"];DICT = 0;
	m = readdlm(string(model_dir,"3DOverthrust801801187.dat"));
	m = m.*1e-3;
	m = reshape(m,801,801,187);
	m = smoothModel3(m,ceil(Int64,801/n[1]));
	Vp = expandModelNearest(m,collect(size(m)),n);m=[];
	# Vp = smoothModel3(Vp,50);
	pad_down = 16;
	pad_down = div(n[3]+pad_down,8)*8 - n[3];
	n_new = n + [0;0;pad_down];
	rho = getLinearModel(2.0,3.0,n);
	
	# Vptag = Vp[:,ceil(Int64,n[2]/4),:];
	# figure();
	# plotModel(Vptag);
	
	T = zeros(tuple(n_new...));
	Tr = zeros(tuple(n_new...));
	T[:,:,1:size(Vp,3)] = Vp;
	Tr[:,:,1:size(Vp,3)] = rho;
	for k=0:pad_down-1
		T[:,:,end-k] = Vp[:,:,end];
		Tr[:,:,end-k] = rho[:,:,end];
	end
	rho = Tr;
	Vp = T;
	Vs = 0.5*Vp;
	rho = 0.25*Vp .+ 1.2;
	
	M = getRegularMesh(domain,n);
	
	# figure();
	# plotModel(Vp,true,M);
	# title("Pressure Velocity");
	# figure();
	# plotModel(Vs,true,M);
	# title("Shear Velocity");
	# figure();
	# plotModel(rho,true,M);
	# title("Density");
	
	mu = rho.*Vs.^2;
	lambda = rho.*(Vp.^2 - 2*Vs.^2);
	
	# println(minimum(mu))
	# println(maximum(mu))
	# println(minimum(lambda))
	# println(maximum(lambda))
	# println(minimum(rho))
	# println(maximum(rho))

	poisson = (lambda./(2*(lambda+mu)));
	# figure();
	# plotModel(poisson,true,M);
	# title("Poisson's ratio");	
	
	domain[end] = domain[end] + pad_down*M.h[3];
	M = getRegularMesh(domain,n_new);
	# rho    = getLinearModel(1.7,2.7,M.n);
	# lambda = getLinearModel(4.2,48.6,M.n);
	# mu     = 0.5*getLinearModel(2.1,24.3,M.n) + 0.5*mu;
end
return rho,lambda,mu,M
end


function getLinearModel(top,bottom,n)
	if length(n)==2
		rho2 = collect(range(top,stop=bottom,length=n[2]));
		rho1 = ones(n[1]);
		rho = rho1*rho2'
	else
		rho_t = collect(range(top,stop=bottom,length=n[2]));
		rho = ones(tuple(n...));
		for k=1:n[3]
			rho[:,:,k].*=rho_t[k];
		end
	end
	return rho;
end

function expandModelNearest(m,n,ntarget)
	if length(size(m))==2
		mnew = zeros(Float64,ntarget[1],ntarget[2]);
		for j=1:ntarget[2]
			for i=1:ntarget[1]
				jorig = convert(Int64,ceil((j/ntarget[2])*n[2]));
				iorig = convert(Int64,ceil((i/ntarget[1])*n[1]));
				mnew[i,j] = m[iorig,jorig];
			end
		end
	elseif length(size(m))==3
		mnew = zeros(Float64,ntarget[1],ntarget[2],ntarget[3]);
		for k=1:ntarget[3]
			for j=1:ntarget[2]
				for i=1:ntarget[1]
					korig = max(1,convert(Int64,floor((k/ntarget[3])*n[3])));
					jorig = convert(Int64,floor((j/ntarget[2])*n[2]));
					iorig = convert(Int64,floor((i/ntarget[1])*n[1]));
					mnew[i,j,k] = m[iorig,jorig,korig];
				end
			end
		end
	end
	return mnew
end


function smoothModel(m,Mesh,times = 0)
	# ms = addAbsorbingLayer2D(m,times);
	ms = copy(m)
	for k=1:times
		for j = 2:size(ms,2)-1
			for i = 2:size(ms,1)-1
				@inbounds ms[i,j] = (2*ms[i,j] + (ms[i-1,j-1]+ms[i-1,j]+ms[i-1,j+1]+ms[i,j-1]+ms[i,j+1]+ms[i+1,j-1]+ms[i+1,j]+ms[i,j+1]))/10.0;
			end
		end
	end
	return ms[(times+1):(end-times),1:end-times];
end

function smoothModel3(m,Mesh,times = 0)
	# ms = addAbsorbingLayer2D(m,times);
	ms = copy(m)
	mt = copy(m)
	n = size(m);
	for t=1:times
		for k = 1:n[3]
			km1 = max(k-1,1);
			kp1 = min(k+1,n[3]);
			for j = 1:n[2]
				jm1 = max(j-1,1);
				jp1 = min(j+1,n[2]);
				for i = 1:n[1]
					im1 = max(i-1,1);
					ip1 = min(i+1,n[1]);
					@inbounds mt[i,j,k] = (2*ms[i,j,k] + (ms[im1,j,k] + ms[i,jm1,k] + ms[i,j,km1] + ms[ip1,j,k] + ms[i,jp1,k] + ms[i,j,kp1]))/8.0;
				end
			end
		end
		ms[:] = mt[:];
	end
	return ms;
end