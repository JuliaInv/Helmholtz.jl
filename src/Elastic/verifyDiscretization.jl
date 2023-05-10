include("ndgrid.jl")
include("elasticFaces.jl")

n = [10,10,10];
h = 1.0./n;


a = [0.5;0.56;0.34];
uj(j,x) = a[j]*sin(pi*x[:,1]).*sin(pi*x[:,2]).*sin(pi*x[:,3]);

mu(X) = sin(0.1*pi*X[:,1]).*sin(0.1*pi*X[:,2]).*sin(0.1*pi*X[:,3]);

function DijUj(i,j,x)
	if i == 1
		return a[j]*pi*cos(pi*x[:,1]).*sin(pi*x[:,2]).*sin(pi*x[:,3]);
	elseif i==2
		return a[j]*pi*cos(pi*x[:,2]).*sin(pi*x[:,1]).*sin(pi*x[:,3]);
	else
		return a[j]*pi*cos(pi*x[:,3]).*sin(pi*x[:,1]).*sin(pi*x[:,2]);
	end
end


nodes(h) = collect(0:h:1.0);
cells(h) = collect((0.5*h):h:(1.0-0.5*h));


# the place where u1 lives:
x1 = nodes(h[1]);
y1 = cells(h[2]);
z1 = cells(h[3]);
(X1,Y1,Z1) = ndgrid(x1,y1,z1);
X1 = [X1[:] Y1[:] Z1[:]];



# the place where u2 lives:
x2 = cells(h[1]);
y2 = nodes(h[2]);
z2 = cells(h[3]);
(X2,Y2,Z2) = ndgrid(x2,y2,z2); 
X2 = [X2[:] Y2[:] Z2[:]];

# the place where u3 lives:
x3 = cells(h[1]);
y3 = cells(h[2]);
z3 = nodes(h[3]);
(X3,Y3,Z3) = ndgrid(x3,y3,z3); 
X3 = [X3[:] Y3[:] Z3[:]];

U1 = uj(1,X1);
U2 = uj(2,X2);
U3 = uj(3,X3);
U = [U1[:];U2[:];U3[:]]


# the place where D11U1 lives:
x = cells(h[1]);
y = cells(h[2]);
z = cells(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D11U1 = DijUj(1,1,X);
M11   = mu(X);


# the place where D12U2 lives:
x = nodes(h[1]);
y = nodes(h[2]);
z = cells(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D12U2 = DijUj(1,2,X);
M12   = mu(X);


# the place where D13U3 lives:
x = nodes(h[1]);
y = cells(h[2]);
z = nodes(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D13U3 = DijUj(1,3,X);
M13   = mu(X);


# the place where D21U1 lives:
x = nodes(h[1]);
y = nodes(h[2]);
z = cells(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D21U1 = DijUj(2,1,X);
M21   = mu(X);


# the place where D22U2 lives:
x = cells(h[1]);
y = cells(h[2]);
z = cells(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D22U2 = DijUj(2,2,X);
M22   = mu(X);



# the place where D23U3 lives:
x = cells(h[1]);
y = nodes(h[2]);
z = nodes(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D23U3 = DijUj(2,3,X);
M23   = mu(X);


# the place where D31U1 lives:
x = nodes(h[1]);
y = cells(h[2]);
z = nodes(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D31U1 = DijUj(3,1,X);
M31   = mu(X);


# the place where D32U2 lives:
x = cells(h[1]);
y = nodes(h[2]);
z = nodes(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D32U2 = DijUj(3,2,X);
M32   = mu(X);


# the place where D32U2 lives:
x = cells(h[1]);
y = cells(h[2]);
z = cells(h[3]);
(X,Y,Z) = ndgrid(x,y,z);
X = [X[:] Y[:] Z[:]];
D33U3 = DijUj(3,3,X);
M33   = mu(X);

GRADU = [D11U1;0.5*(D21U1 + D12U2) ; 0.5*(D31U1 + D13U3);0.5*(D21U1 + D12U2) ; D22U2 ; 0.5*(D23U3 + D32U2) ; 0.5*(D31U1 + D13U3) ; 0.5*(D23U3 + D32U2) ; D33U3 ];

GRAD,DIV,GG,D11,D12,D13,D21,D22,D23,D31,D32,D33 = getDifferentialOperators(n,h);

GRADhU = GRAD*U;
E = GRADhU - GRADU;
println(norm(E,Inf));

println(norm(D11*U1 - D11U1 ,Inf));
println(norm(D12*U2 - D12U2 ,Inf));
println(norm(D13*U3 - D13U3 ,Inf));
println(norm(D21*U1 - D21U1 ,Inf));
println(norm(D22*U2 - D22U2 ,Inf));
println(norm(D23*U3 - D23U3 ,Inf));
println(norm(D31*U1 - D31U1 ,Inf));
println(norm(D32*U2 - D32U2 ,Inf));
println(norm(D33*U3 - D33U3 ,Inf));

####################################################################################

println("Mass")
Mnum = diag(getMassMatrix(M33,n));
Mtrue = [M11;M12;M13;M21;M22;M23;M31;M32;M33];

println(norm(Mnum-Mtrue,1)/length(Mnum))
println(norm(Mnum-Mtrue,Inf))

