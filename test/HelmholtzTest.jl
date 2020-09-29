using jInv.Mesh;
using Helmholtz

const plotting = false;
if plotting
	using PyPlot;
	close("all")
end


# m = readdlm("SEGmodel2Dsalt.dat"); m = m';
m = ones(257,129);
m = m*1e-3;
Minv = getRegularMesh([0.0,13.5,0.0,4.2],collect(size(m)) .- 1);

pad = 25;

m = ones(size(m));
m = 1.0./m.^2

if plotting
	figure();
	imshow(1.0./sqrt.(Matrix(m')))
end
f = 1.0;
w = 2*pi*f

maxOmega = getMaximalFrequency(m,Minv);
ABLamp = maxOmega;

println("omega*h:");
println(w*Minv.h*sqrt(maximum(m)));
pad = pad*ones(Int64,Minv.dim);
H = GetHelmholtzOperator(Minv,m,w,ones(size(m))*0.01,true,pad,ABLamp,true)[1];
SH = H + GetHelmholtzShiftOP(m, w,0.1);

n_nodes = Minv.n.+1;
nnodes_tup = tuple(n_nodes...);

q,src = getAcousticPointSource(Minv,ComplexF64);

s = H\q[:];
s = real(reshape(s,nnodes_tup));
s[loc2cs(n_nodes,src)] = 0.0;

if plotting
	figure();
	imshow(s');title("Helmholtz Solution");
end


sh = SH\q[:];
sh = real(reshape(sh,nnodes_tup));
sh[loc2cs(n_nodes,src)]= 0.0;

if plotting
	figure();
	imshow(sh');title("Shifted Helmholtz Solution");
end