using jInv.Mesh;
using ForwardHelmholtz
using ForwardHelmholtz.PointSourceADR

plotting = true;

if plotting
	using PyPlot;
	# close("all");
end

include("getModels.jl");

NeumannAtFirstDim = true;
Sommerfeld = true;
gamma_val = 0.00001*2*pi;

# model = "const";
# model = "linear";
# model = "linear squared slowness";
# model = "Gaussian"
# model = "Wave-guide"
model = "Wedge"
domain = [0.0,1.0,0.0,1.0];
	n_cells = [512,512];
src = div(n_cells,2)+1;
src[2] = 1;

if model=="const"
	NeumannAtFirstDim = false;
	Sommerfeld = true;
	f = 40.0;
	src = div(n_cells,2)+1;
	m_coarse = getConstModel(n_cells+1);
	pad_cells = [20;20];
elseif model=="linear"
	f = 40.0
	m_coarse = getLinearModel(1.6,3.5,n_cells+1);m_coarse = 1./m_coarse.^2
	pad_cells = [20;20];
elseif model=="linear squared slowness"
	f = 70.0
	m_coarse = getInvLinearModel(1.6,3.5,n_cells+1);
	pad_cells = [20;20];
elseif model=="Gaussian"
	f = 40.0
	pad_cells = [50;50];
	Minv = getRegularMesh(domain,n_cells);
	m_coarse = getSmoothGaussianMedium(n_cells+1,Minv.h)[1];
elseif model=="Wave-guide"
	pad_cells = [64;64];	
	f = 20.0; 
	m_coarse = waveGuideModel(n_cells+1)[1];
elseif model=="Wedge"
	f = 20.0;
	pad_cells = [36;36];
	n_cells = [512,512];  (m_coarse,Minv) = getWedgeModel(n_cells+1);domain = Minv.domain;
else
	error("unknown model");
end
Minv = getRegularMesh(domain,n_cells);

# m_coarse = getLinearModel(1.6,3.5,n_cells+1);
# m_coarse = 1./m_coarse.^2
# m_coarse = getLinearModel(1./((1.6)^2),1./(3.5)^2,n_cells+1);
# m_coarse = getSmoothGaussianMedium(n_cells+1,Minv.h)[1];
# 

w = 2*pi*f

figure()
imshow(m_coarse'); colorbar();
title(string("The ",model," model"))
# error("Here")


n_tup = tuple((n_cells+1)...);
q_coarse = zeros(Complex,n_tup)
q_coarse[src[1],src[2]] = 1/(Minv.h[2]^2);

# src2 = div(3*n_cells,4)+1;
# q_coarse[src2[1],src2[2]] = 1/(Minv.h[2]^2);
# src = src2;


println("omega*h:");
println(w*Minv.h*sqrt(maximum(m_coarse)));

# m = readdlm("SEGmodel2Dsalt.dat"); m = m'; m = m*1e-3;
maxOmega = getMaximalFrequency(m_coarse,Minv);
ABLamp = maxOmega;


### GENERATE FINE REFERENCE ##############################################################################
factor = 4;
n_cells_fine = factor*n_cells;
pad_cells_fine = factor*pad_cells;

Minv_fine = getRegularMesh(domain,n_cells_fine);
src_fine = div(n_cells_fine,2)+1;
if src[2]==1 
	src_fine[2] = 1;
end
n_tup = tuple((n_cells_fine+1)...);
q = zeros(Complex,n_tup)
q[src_fine[1],src_fine[2]] = 1/(Minv_fine.h[2]^2);

# src2 = div(3*n_cells_fine,4)+1;
# q[src2[1],src2[2]] = 1/(Minv_fine.h[2]^2);

m=0;
if model=="const"
	m = getConstModel(n_cells_fine+1);
elseif model=="linear"
	m = getLinearModel(1.6,3.5,n_cells_fine+1);m = 1./m.^2
elseif model=="linear squared slowness"
	m = getInvLinearModel(1.6,3.5,n_cells_fine+1);
elseif model=="Gaussian"
	m = getSmoothGaussianMedium(n_cells_fine+1,Minv_fine.h)[1];
elseif model=="Wave-guide"
	m = waveGuideModel(n_cells_fine+1)[1];
elseif model=="Wedge"
	m = getWedgeModel(n_cells_fine+1)[1];
end
# 
# 

H, = GetHelmholtzOperator(Minv_fine, m, w, gamma_val*ones(Float32,size(m)),NeumannAtFirstDim,pad_cells_fine+1,2.0*ABLamp,Sommerfeld);
xh_fine = (H\q[:]);
xh_fine = reshape(xh_fine,n_tup);

n_tup = tuple(n_cells+1...);
xh_ref = zeros(Complex128,n_tup);
xh_ref = xh_fine[1:factor:end,1:factor:end];
Iref = xh_ref
figure()
imshow(real(xh_ref)');colorbar();
### FINE REFERENCE DONE ##############################################################################

m = m_coarse;

gamma = gamma_val*ones(Float32,size(m)) + getABL(Minv,NeumannAtFirstDim,pad_cells+1,ABLamp);
(ADR_long,ADR_short,T) = getHelmholtzADR(true,Minv, m, w, gamma,NeumannAtFirstDim,src,Sommerfeld);
(ADR_longUp,ADR_short,T) = getHelmholtzADR(false,Minv, m, w, gamma,NeumannAtFirstDim,src,Sommerfeld);



figure();
imshow(real(exp(1im*w*reshape(T,n_tup)))');

H, = GetHelmholtzOperator(Minv, m, w, gamma_val*ones(Float32,size(m)),NeumannAtFirstDim,pad_cells+1,ABLamp,Sommerfeld);
S =  GetHelmholtzShiftOP(m, w,0.2);

q = q_coarse;
xh = (H\q[:])
q = q[:].*exp(1im*w*T);
xl = (ADR_long\q[:]).*exp(-1im*w*T);
xs = ((ADR_short )\q[:]).*exp(-1im*w*T);
us = (H+S)\q[:];
alu = (ADR_longUp\q[:]);
xlu = alu.*exp(-1im*w*T);





I1 = reshape(xh,n_tup);
I2 = reshape(xl,n_tup); 
I3 = reshape(xlu,n_tup);
I4 = reshape(xs,n_tup);
I5 = reshape(us,n_tup);


b = maximum(real([I1[:];I2[:];I3[:]]));
a = minimum(real([I1[:];I2[:];I3[:]]));
a_err = 1.0;
N = abs(Iref);
N = N + 0.1*mean(N);
# N[:]=1.0;

E1 = abs(Iref-I1)./N;
E2 = abs(Iref-I2)./N;
E3 = abs(Iref-I3)./N;
E4 = abs(Iref-I4)./N;
# E1[src[1],src[2]] = 0.0;
# E2[src[1],src[2]] = 0.0;
# E3[src[1],src[2]] = 0.0;
# E4[src[1],src[2]] = 0.0;

if plotting
figure();
subplot(1,4,1);
imshow(real(I2'));
subplot(1,4,2);
imshow(reshape(T,n_tup)');
subplot(1,4,3);
imshow(reshape(real(alu),n_tup)');
subplot(1,4,4);
imshow(reshape(real(exp(-1im*w*T)),n_tup)');




figure();
subplot(1,3,1);
imshow(real(I1')); title("Full Helmholtz solution")
subplot(1,3,2);
imshow(reshape(real(us),n_tup)'); title("Shifted Helmholtz solution")
subplot(1,3,3);
imshow(reshape(real(xs),n_tup)'); title("First order upwind ADR solution")




# figure()
# subplot(2,4,1);
# imshow(real(I1'),clim = [a,b]); colorbar();title("Standard discretization")
# subplot(2,4,2);
# imshow(real(I2'),clim = [a,b]);colorbar();title("ADR discretization - central")
# subplot(2,4,3);
# imshow(real(I3'),clim = [a,b]);colorbar();title("ADR discretization - 2nd upwind")
# subplot(2,4,4);
# imshow(real(I4'),clim = [a,b]);colorbar();title("ADR discretization - 1st upwind")

# subplot(2,4,5);
# imshow(E1',clim = [0.0,a_err]);colorbar();title("Relative error - standard")
# subplot(2,4,6);
# imshow(E2',clim = [0.0,a_err]);colorbar();title("Relative error - ADR central")
# subplot(2,4,7);
# imshow(E3',clim = [0.0,a_err]);colorbar();title("Relative error - ADR upwind")
# subplot(2,4,8);
# imshow(E4',clim = [0.0,a_err]);colorbar();

figure()
subplot(2,3,1);
imshow(real(I1'),clim = [a,b]); colorbar();title("Standard discretization"); 
subplot(2,3,2);
imshow(real(I2'),clim = [a,b]);colorbar();title("ADR discretization - central")
subplot(2,3,3);
imshow(real(I3'),clim = [a,b]);colorbar();title("ADR discretization - 2nd order upwind")

subplot(2,3,4);
imshow(E1',clim = [0.0,a_err]);colorbar();title("Relative error - standard")
subplot(2,3,5);
imshow(E2',clim = [0.0,a_err]);colorbar();title("Relative error - ADR central")
subplot(2,3,6);
imshow(E3',clim = [0.0,a_err]);colorbar();title("Relative error - ADR upwind")





# figure()
# subplot(2,3,1);
# imshow(reshape(T,n[1],n[2])');colorbar();
# subplot(2,3,2);
# imshow(reshape(real(exp(1im*w*T)),n[1],n[2])'); colorbar();
# subplot(2,3,3);
# imshow(reshape(G1,n[1],n[2])');colorbar();
# subplot(2,3,4);
# imshow(reshape(G2,n[1],n[2])');colorbar();
# subplot(2,3,5);
# imshow(reshape(G1.^2 + G2.^2,n[1],n[2])');colorbar();
# subplot(2,3,6);
# imshow(reshape(LT,n[1],n[2])');colorbar();

end

error("ET")