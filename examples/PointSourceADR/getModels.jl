import FactoredEikonalFastMarching.getAnalytic3DeikonalSolutionAll
import FactoredEikonalFastMarching.getAnalytic2DeikonalSolutionAll

function getConstModel(n)
m = ones(Float64,tuple(n...));
return m;
end

function getInvLinearModel(top,bottom,n)
	return getLinearModel(1.0./(top^2),1.0./(bottom^2),n);;
end


function getLinearModel(top,bottom,n)
	if length(n)==2
		rho2 = linspace(top,bottom,n[2]);
		rho1 = ones(Float64,n[1]);
		rho = rho1*rho2'
	else
		rho_t = linspace(top,bottom,n[3]);
		rho = ones(Float64,tuple(n...));
		for k=1:n[3]
			rho[:,:,k].*=rho_t[k];
		end
	end
	return rho;
end

function getWedgeModel(n::Array{Int64});
h = 1.0./(n-1);
x = linspace(0.0,1.0,n[1]);
y = linspace(0.0,1.0,n[2]);

X = x*ones(Float64,n[2])';
Y = ones(Float64,n[1])*y';
Z = 0.25*(tanh.((4*Y-X - 0.75)*20)) + 0.75; 
Z[:,end - div(n[2],2)+1:end] = Z[:,div(n[2],2) : -1 : 1];
return Z.^2,getRegularMesh([0.0,1.0,0.0,1.0],n-1);
end

function waveGuideModel(n::Array{Int64})
h = 1.0./(n-1);
x = linspace(0.0,1.0,n[1]);
y = linspace(0.0,1.0,n[2]);

X = x*ones(Float64,n[2])';
Y = ones(Float64,n[1])*y';
Z = 1.25*(1-0.4*exp.(-32*abs(X - 0.5).^2));
return 1.0./(Z.^2),getRegularMesh([0.0,1.0,0.0,1.0],n-1);
end

function getSmoothGaussianMedium(n::Array{Int64,1},h::Array{Float64,1})

src_kappa = zeros(Int64,2);
xsrc = zeros(Float64,2);
src_kappa[1] = div(n[1],2);
src_kappa[2] = div(n[2],2);

xsrc[1] = (src_kappa[1]-1)*h[1];
xsrc[2] = (src_kappa[2]-1)*h[2];

X1,X2 = ndgrid((0:(n[1]-1))*h[1],(0:(n[2]-1))*h[2]);

kappaSquared = exp.(-4*(X1 - xsrc[1]).^2 - 8*(X2 - xsrc[2]).^2);
return kappaSquared,1;
end

# function getSmoothGaussianMedium(n::Array{Int64,1},h::Array{Float64,1})
# src = div(n,2);
# kappaSquared = [];
# T_exact = [];
# if length(n)==2
	# (T1_exact,G11_exact,G12_exact) = getSmoothFactoredModel(n,h);
	# (T0,G01,G02,L0) = getAnalytic2DeikonalSolutionAll(n,h,src);
	# G1_exact = T0.*G11_exact + G01.*T1_exact;
	# G2_exact = T0.*G12_exact + G02.*T1_exact;
	# kappaSquared = G1_exact.*G1_exact + G2_exact.*G2_exact;
	# T_exact = T0.*T1_exact;
# else
	# (T1_exact,G11_exact,G12_exact,G13_exact) = getSmoothFactoredModel3D(n,h);
	# (T0,G01,G02,G03) = getAnalytic3DeikonalSolutionAll(n,h,src);
	# G1_exact = T0.*G11_exact + G01.*T1_exact;
	# G2_exact = T0.*G12_exact + G02.*T1_exact;
	# G3_exact = T0.*G13_exact + G03.*T1_exact;
	# kappaSquared = G1_exact.*G1_exact + G2_exact.*G2_exact + G3_exact.*G3_exact;
	# T_exact = T0.*T1_exact;
# end

# return (kappaSquared),src,T_exact;
# end

function getSmoothFactoredModel(n::Array{Int64,1},h::Array{Float64,1})
xsrc = zeros(2);
src_kappa = zeros(Int64,2);
src_kappa[1] = div(n[1],2);
src_kappa[2] = div(n[2],2);

xsrc[1] = (src_kappa[1]-1)*h[1];
xsrc[2] = (src_kappa[2]-1)*h[2];

X1,X2 = ndgrid((0:(n[1]-1))*h[1],(0:(n[2]-1))*h[2]);

sigma = 1.0;
T1_exact = (exp.( - (sigma*((X1 - xsrc[1]).^2) + 4*sigma*((X2-xsrc[2]).^2))) + 1)/2;
G11_exact = -2*sigma*(X1 - xsrc[1]).*exp( - (sigma*((X1 - xsrc[1]).^2) + 4*sigma*((X2-xsrc[2]).^2)))/2;
G12_exact = -8*sigma*(X2 - xsrc[2]).*exp( - (sigma*((X1 - xsrc[1]).^2) + 4*sigma*((X2-xsrc[2]).^2)))/2;

return T1_exact,G11_exact,G12_exact;
end









# function getSmoothFactoredModel3D(n::Array{Int64,1},h::Array{Float64,1})
# xsrc = zeros(3);
# src_kappa = zeros(Int64,3);
# src_kappa[1] = div(n[1],3);
# src_kappa[2] = div(n[2],4);
# src_kappa[3] = div(n[3],2);

# xsrc[1] = (src_kappa[1]-1)*h[1];
# xsrc[2] = (src_kappa[2]-1)*h[2];
# xsrc[3] = (src_kappa[3]-1)*h[3];

# X1,X2,X3 = ndgrid((0:(n[1]-1))*h[1],(0:(n[2]-1))*h[2],(0:(n[3]-1))*h[3]);

# Sigma = [0.1,0.4,0.2];
# EXPRSkewedSquared = exp(-Sigma[1]*((X1 - xsrc[1]).^2) - Sigma[2]*((X2-xsrc[2]).^2) - Sigma[3]*((X3-xsrc[3]).^2));

# T1_exact = EXPRSkewedSquared/2 + 1/2;
# G11_exact = -Sigma[1]*(X1 - xsrc[1]).*EXPRSkewedSquared;
# G12_exact = -Sigma[2]*(X2 - xsrc[2]).*EXPRSkewedSquared;
# G13_exact = -Sigma[3]*(X3 - xsrc[3]).*EXPRSkewedSquared;

# return T1_exact,G11_exact,G12_exact,G13_exact;
# end
