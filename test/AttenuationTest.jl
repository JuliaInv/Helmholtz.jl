using jInv.Mesh;
using Helmholtz
import FactoredEikonalFastMarching.getAnalytic2DeikonalSolution
const plotting = false;
if plotting
	using PyPlot;
	close("all");
end

n_tup = (512,512)
m = ones(n_tup);
n = collect(n_tup)
Minv = getRegularMesh([0.0,10.0,0.0,10.0],n.-1);

pad = 35;
f = 0.5;
fmax = 2.5;
f = fmax;
w = 2*pi*f

src = div.(n,2);
q = zeros(ComplexF64,n_tup)
q[loc2cs(n,src)] = 1.0./(Minv.h[1]^2);


r = getAnalytic2DeikonalSolution(n,Minv.h,src)[1];
a = 1.0./(sqrt.(r.+Minv.h[1]));
s_analytic = real(exp.(1im*pi/4).*a.*exp.(1im*w.*m.*r));


alpha = 0.5*2*pi;
maxOmega = getMaximalFrequency(m,Minv);
ABLamp = maxOmega;

gamma = getABL(n,false,[pad,pad],ABLamp);


H = GetHelmholtzOperator(Minv, m, w, gamma.+alpha, false,false);

s1 = H\q[:];
s1 = real(reshape(s1,n_tup));

H = 0;
H = GetHelmholtzOperator(Minv, m, w-1im*alpha./2.0, gamma[:], false,false);

s2 = H\q[:];
s2 = real(reshape(s2,n_tup));


s_analytic = s_analytic./maximum(s_analytic);
s_analytic.*=exp.(-(alpha/2).*m.*r)
s_analytic.*=maximum(s1);

if plotting
	figure();
	imshow(s1');title("Helmholtz Solution");
	figure();
	imshow(s_analytic');title("Analytic Solution");
end

if plotting
	figure();
	plot(s1[src[1],:],"b");
	plot(s2[src[1],:],"r");
	plot(s_analytic[src[1],:],"g");
	legend(("s1","s2","Analytic"));
	
end



