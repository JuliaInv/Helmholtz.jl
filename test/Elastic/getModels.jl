using DelimitedFiles

function getModel(model, n, lambda_param = 16.0, mu_param = 1.0, rho_param = 1.0; stepnum = 8)
	if model == "SEG"
		domain = [0.0,4.2,0.0,13.5];

		if length(n) == 3
			println("SEG is a 2D model")
		end

		model_dir = "BenchmarkModels/";
		Vp = readdlm(string(model_dir,"SEGmodel2Dsalt.dat"))/1000.0;

		n_data = collect(size(Vp));
		Vp = expandModelNearest(smoothModel(Vp,[],0),n_data,n);
		Vs = 0.5 .* Vp;
		rho = 0.25 .* Vp .+ 1.5;
		mu = rho .* Vs .^ 2;
		lambda = rho .* Vp .^2 .- 2 .* mu;
		M = getRegularMesh(domain,n);


	elseif model == "wedge"

		rho = getWedge(2.0,3.0,n);
		lambda = getWedge(4.0,20.0,n);
		mu = getWedge(1.0,15.0,n);

		domain = [0.0;5.0;0.0;5.0];
		M = getRegularMesh(domain,n);

	elseif model == "zebra"
		domain = [0.0,16.0,0.0,5.0];
		if length(n) == 3
			domain = [0.0,16.0,0.0,16.0,0.0,5.0];
		end

		rho    = getZebra(2.0,3.0,n,stepnum);
		lambda = getZebra(4.0,20.0,n,stepnum);
		mu     = getZebra(1.0,15.0,n,stepnum);

		Vs = sqrt.(mu./rho);
		Vp = sqrt.((lambda+2*mu)./rho)
		Poisson = (lambda./(2*(lambda+mu)));
		M = getRegularMesh(domain,n);
		
	elseif model == "linear"
		domain = [0.0,16.0,0.0,5.0];
		if length(n) == 3
			domain = [0.0,16.0,0.0,16.0,0.0,5.0];
		end

		rho    = getLinearModel(2.0,3.0,n);
		lambda = getLinearModel(4.0,20.0,n);
		mu     = getLinearModel(1.0,15.0,n);
			
		Vs = sqrt.(mu./rho);
		Vp = sqrt.((lambda+2*mu)./rho)
		Poisson = (lambda./(2*(lambda+mu)));

		M = getRegularMesh(domain,n);

	elseif model=="Marmousi"
		if length(n)==3
			error("This code supports only 2D Marmousi");
		end

		domain = [0.0,17.0,0.0,3.5]; ## that's the original domain
		# model_dir = "./../../../BenchmarkModels/";
		# model_dir = "./../BenchmarkModels/";
		model_dir = "BenchmarkModels/";
		
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

		n_data = collect(size(Vs));
		
		mu = rho.*Vs.^2
		lambda = rho.*(Vp.^2 - 2*Vs.^2)
		poisson = (lambda./(2*(lambda+mu)));
		
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
		poisson = (lambda./(2*(lambda+mu)));	

		domain_data = [0.0,17.0,0.0,3.0 + pad_down*M.h[2]];

		n_data = collect(size(rho));
		M = getRegularMesh(domain_data,n_data);
	
	elseif model=="SEAM"

		# I first cutted the 162 first rows which contain sea, 
		# then saved it as a matlab variable, then used
		# save SEAMrhosmall.dat SEAMrhosmall -ascii
		# to save it in a format julia can read
		
		if length(n)==3
			error("SEAM is 2D");
		end
		domain = [0.0,13.5,0.0,35.0]; ## that's the original domain

		model_dir = "BenchmarkModels/";
		
		println("Reading SEAM")
		Vs = readdlm(string(model_dir,"SEAMvssmall.dat"))/1000.0;
		println("1")
		Vp = readdlm(string(model_dir,"SEAMvpsmall.dat"))/1000.0;
		println("2")
		rho = readdlm(string(model_dir,"SEAMrhosmall.dat"));
		println("Finished reading SEAM")
		
		n_data = collect(size(Vs));
		M = getRegularMesh(domain,n_data);
		
		Vs = expandModelNearest(smoothModel(Vs,[],0),n_data,n);
		Vp = expandModelNearest(smoothModel(Vp,[],0),n_data,n);
		rho = expandModelNearest(smoothModel(rho,[],0),n_data,n);
				
		mu = rho.*Vs.^2;
		lambda = rho.*(Vp.^2 - 2*Vs.^2);
		poisson = (lambda./(2*(lambda+mu)));

		M = getRegularMesh(domain,n);

	elseif model=="const"

		n_tup = tuple(n...);
		rho    = rho_param * ones(n_tup);
		lambda = lambda_param * ones(n_tup);
		mu     = mu_param * ones(n_tup);
		domain = [0.0,16.0,0.0,5.0];
		if length(n) == 3
			domain = [0.0,16.0,0.0,16.0,0.0,5.0];
		end
		M = getRegularMesh(domain,n);

	elseif model == "Overthrust" || model == "OverthrustUmair" || model == "Overthrust025" || model == "Overthrust045" 

		domain = [0.0,20.0,0.0,20.0,0.0,4.65];
		# model_dir = "./../../../BenchmarkModels/";
		# model_dir = "./../BenchmarkModels/";
		model_dir = "BenchmarkModels/";

		m = readdlm(string(model_dir,"3DOverthrust801801187.dat"));
		m = m .* 1e-3;
		m = reshape(m,801,801,187);
		m = smoothModel(m,ceil(Int64,801/n[1]));
		Vp = expandModelNearest(m,collect(size(m)),n);m=[];
		pad_down = 16;
		pad_down = div(n[3]+pad_down,8)*8 - n[3];
		n_new = n + [0;0;pad_down];
		rho = getLinearModel(2.0,3.0,n);
		
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

		if model == "Overthrust" || model == "Overthrust025" || model == "Overthrust045" 
			if model == "Overthrust" 
				VpVsRatio = 2;
			elseif model == "Overthrust025"
				VpVsRatio = 1.75;
			elseif model == "Overthrust045"
				VpVsRatio = 3.5;
			end
			Vs = (1/VpVsRatio) * Vp;
			rho = 0.25 * Vp .+ 1.2;
		elseif model == "OverthrustUmair"
			### SI units ###
			# Vp = 1000 .* Vp;
			# rho = 310 * Vp .^ (0.25);
			# range = [1090.4; 3529.4];
			### gr/cm^3 and km/sec ###
			rho = 1.74 * Vp .^ (0.25);
			range = [1090.4; 3529.4] ./ 10^3;
			dr = range[2] - range[1];
			Vs = range[1] .+ ((Vp .- minimum(Vp))*dr) ./ (maximum(Vp) - minimum(Vp));
		end
		mu = rho .* Vs .^ 2;
		lambda = rho .* (Vp .^ 2 - 2 * Vs .^ 2);
		
		M = getRegularMesh(domain,n);
		
		mu = rho.*Vs.^2;
		lambda = rho.*(Vp.^2 - 2*Vs.^2);

		Vs = 0.0;
		Vp = 0.0;
		T = 0.0;
		Tr = 0.0;
		
		poisson = (lambda./(2*(lambda+mu)));
		
		domain[end] = domain[end] + pad_down*M.h[3];
		M = getRegularMesh(domain,n_new);

	end
	return rho,lambda,mu,M
end


function getWedge(bottom, top, n);

	if length(n) == 3
		println("This code supports only 2D wedge media")
	elseif ~ (n[1] == n[2])
		println("n must be square")
	end

    nx = n[1]; ny = n[1];
    x = (0:nx - 1) ./ (nx - 1);
    y = (0:ny - 1) ./ (ny - 1);  
    X = x * ones(Float64, nx)';
    Y = ones(Float64,nx) * y';

    Z = 0.25 .* (tanh.((4 .* Y - X .- 0.75) .* 20)) .+ 0.75; 
    Z[:, end - div(ny, 2) + 1:end] = Z[:, div(ny, 2) : -1 : 1];
    
    ratios = Z .^ 2;

    ratios = ratios .* ((top - bottom) / (1 - 0.25)); # stretch
	top_temp = 1 * ((top - bottom) / (1 - 0.25));
	ratios = ratios .+ (top - top_temp); # translation

    return ratios;
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

# add if on size(m) for dimension and make the two functions into one

function smoothModel(m,Mesh,times = 0)
	
	if length(size(m)) == 2
		# ms = addAbsorbingLayer2D(m,times);
		ms = copy(m)
		for k=1:times
			for j = 2:size(ms,2)-1
				for i = 2:size(ms,1)-1
					@inbounds ms[i,j] = (2*ms[i,j] + (ms[i-1,j-1]+ms[i-1,j]+ms[i-1,j+1]+ms[i,j-1]+ms[i,j+1]+ms[i+1,j-1]+ms[i+1,j]+ms[i,j+1]))/10.0;
				end
			end
		end
		ms = ms[(times+1):(end-times),1:end-times];
	elseif length(size(m)) == 3
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
	end
		
	return ms;
end
