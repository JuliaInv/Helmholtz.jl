using jInv.Mesh;
using Helmholtz
using Helmholtz.ElasticHelmholtz
using Statistics
using SparseArrays
using LinearAlgebra

plotting = false;
if plotting
	include("plottingForSystems.jl")
	using PyPlot;
	close("all")
end


dim = 2;
if dim==3
	n = [64,64,32]; n_tup = tuple(n...);
	Minv = getRegularMesh([0.0,14,0.0,14.0,0.0,7.0],n);
	q = getElasticPointSource(Minv,ComplexF64);
	neumann_on_top = true
	rho    = 1.0*ones(n_tup);
	lambda = 1.0*ones(n_tup);
	mu     = 0.5*ones(n_tup);
	pad    = 8;
	f      = 0.5;
else
	n = [512,256];f = 2.5;
	# n = [256,128];f = 1.1;
	# n = [128,64];f = 0.55;
	factor = 1;
	n = n*factor;
	n_tup = tuple(n...);
	Minv = getRegularMesh([0.0,14.0,0.0,7.0],n);
	## Generating the right hand side
	# q = getElasticPointSource(Minv,ComplexF64);
	q = getElasticPointSourceMid(Minv,ComplexF64)
	neumann_on_top = false
	rho    							= 1e-0*ones(n_tup);
	lambda 							= 1e-0*ones(n_tup);
	mu     							= 0.5*ones(n_tup);
	pad 							= 25;
	
end

# if plotting
	# figure();
	# plotVectorU2D(q,Minv,"q")
# end



n_tup = tuple(n...);

sp = sqrt.(rho[:]./(lambda[:] + 2*mu[:]))
ss = sqrt.(rho[:]./(mu[:]))

w = 2*pi*f

println("sp*omega*h = ",w*maximum(Minv.h)*maximum(sp));
println("ss*omega*h = ",w*maximum(Minv.h)*maximum(ss));

pad = pad*ones(Int64,3);

gamma = getCellCenteredABL(Minv,neumann_on_top,pad,2.0*w);
gamma .+= 0.001;	

mixed = false;

Hparam = ElasticHelmholtzParam(Minv,w,lambda,rho,mu,gamma,true,mixed)

H = GetElasticHelmholtzOperator(Hparam);
shift = 0.15;
shift_mass = 1im*(1.0./prod(Minv.h))*shift*(w^2)*rho[:];
Shift = getFaceMassMatrix(Minv, shift_mass, saveMat=false, avN2C = avN2C_Nearest);
SH = H + Shift;

u = H\q;
us = SH\q;
Div = getDivergenceMatrix(Minv);
p = Div*u
p = reshape(p,(Minv.n[1],Minv.n[dim]));
p = real(p);
# if plotting
	# figure();
	# ur = real(u);
	# plotVectorU2D(ur,real(us),Minv,("Original u","Shited u"),clim = [mean(ur[:]) - 4*std(ur[:]), mean(ur[:]) + 4*std(ur[:])], cmap="viridis");
	# p = reshape(p,(Minv.n[1],Minv.n[dim]));
	# figure()
	# p = real(p);
	# imshow(real(p)',cmap="viridis",clim = [mean(p[:]) - 4*std(p[:]), mean(p[:]) + 4*std(p[:])]);colorbar()
	# title("Acoustic (mu = 0) : p = div*u ")
# end

if plotting ### FIGURE 5 in JCP paper
	figure();
	ur = real(u);
	ux = getUjProjMatrix(Minv.n,1)'*u;
	uy = getUjProjMatrix(Minv.n,2)'*u;
	ux = reshape(ux,(Minv.n[1]+1,Minv.n[2]));
	uy = reshape(uy,(Minv.n[1],Minv.n[2]+1));

	#subplot(1,3,1)
	imshow(real(ux)',clim = [mean(ur[:]) - 4*std(ur[:]), mean(ur[:]) + 4*std(ur[:])], cmap = "viridis");#colorbar()
	#axis("off")
	#subplot(1,3,2)
	figure()
	imshow(real(uy)',clim = [mean(ur[:]) - 4*std(ur[:]), mean(ur[:]) + 4*std(ur[:])], cmap = "viridis");#colorbar()
	#axis("off")
	#subplot(1,3,3)
	figure()
	imshow(p',cmap="viridis",clim = [mean(p[:]) - 4*std(p[:]), mean(p[:]) + 4*std(p[:])]);#colorbar()
	#axis("off")
end

# if plotting ### FIGURE 1 in JCP paper
	# figure();
	# ur = real(u);
	# ux = getUjProjMatrix(Minv.n,1)'*u;
	# ux = reshape(ux,(Minv.n[1]+1,Minv.n[2]));

	# usr = real(us);
	# usx = getUjProjMatrix(Minv.n,1)'*us;
	# usx = reshape(usx,(Minv.n[1]+1,Minv.n[2]));


	# #subplot(1,3,1)
	# figure()
	# imshow(real(ux)',clim = [mean(ur[:]) - 4*std(ur[:]), mean(ur[:]) + 4*std(ur[:])], cmap = "viridis");#colorbar()
	# figure()
	# imshow(real(usx)',clim = [mean(ur[:]) - 4*std(ur[:]), mean(ur[:]) + 4*std(ur[:])], cmap = "viridis");#colorbar()
# end
############################################################################################################################
############################################ Reformulation #################################################################
############################################################################################################################

println("****************************  Reformulated equation:  ******************************")

Hparam.MixedFormulation = true; 
Hr = GetElasticHelmholtzOperator(Hparam);
Shift = blockdiag(Shift,spzeros(prod(Minv.n),prod(Minv.n)));

SHr = Hr + Shift;
q = [q;zeros(ComplexF64,prod(Minv.n))];


# # LU = lufact(SHr);
# # Mexact(x) = LU\x; 
# # x, flag,rnorm,iter = KrylovMethods.bicgstb(Hr,q,tol = 1e-5,maxIter = 50,M1 = Mexact,M2 = identity, x = zeros(eltype(q),size(q)),out=2);


ur = Hr\q;
urs = SHr\q;
println(norm(u - ur[1:length(u)]))

println(norm(us - urs[1:length(us)]))


println("****************************  Beta-Reformulated equation:  ******************************")

Hparam.MixedFormulation = true; 
Hrb = GetElasticHelmholtzOperator(Hparam,spread=true,beta=2/3);

u = Hrb\q

if plotting
	figure();
	ur = real(u[1:(end-prod(Minv.n))]);
	ux = getUjProjMatrix(Minv.n,1)'*ur;
	uy = getUjProjMatrix(Minv.n,2)'*ur;
	ux = reshape(ux,(Minv.n[1]+1,Minv.n[2]));
	uy = reshape(uy,(Minv.n[1],Minv.n[2]+1));

	#subplot(1,3,1)
	imshow(real(ux)',clim = [mean(ur[:]) - 4*std(ur[:]), mean(ur[:]) + 4*std(ur[:])], cmap = "viridis");#colorbar()
	#axis("off")
	#subplot(1,3,2)
	title("Spread ux")
	figure()
	imshow(real(uy)',clim = [mean(ur[:]) - 4*std(ur[:]), mean(ur[:]) + 4*std(ur[:])], cmap = "viridis");#colorbar()
	#axis("off")
	#subplot(1,3,3)
	title("Spread uy")
	figure()
	
	p = -u[(end-prod(Minv.n)+1):end]
	p = reshape(real(p),(Minv.n[1],Minv.n[dim]));
	figure()
	imshow(p',cmap="viridis",clim = [mean(p[:]) - 4*std(p[:]), mean(p[:]) + 4*std(p[:])]);#colorbar()
	#axis("off")
end


println("THE END");
















