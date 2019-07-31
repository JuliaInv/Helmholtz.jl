# using PyPlot
# close("all")
using jInv.Mesh
using Helmholtz
using LinearAlgebra
using Test
using Printf


function fictitiousSourceTest2D(u,m,w,rhs)
  refineMesh(M::RegularMesh) = (getRegularMesh(M.domain, M.n*2))

  M  = getRegularMesh([-1.0 1.0 -1.0 1.0],[32,48])
  N  = 4
  err = zeros(N,2)
  # k=3
  uk = 0.0; sk= 0.0; rk = 0.0;k=1;ut=0
   for k=1:N
      M  = refineMesh(M)
  	  xn = getNodalGrid(M)
	  
	  gamma = getABL(M,true,5*ones(Int64,M.dim),0.1);

  	  uk = u(xn[:,1],xn[:,2])
      mk = m(xn[:,1],xn[:,2]);
	  
      rk = rhs(xn[:,1],xn[:,2]) .+ 1im*(w^2).*gamma[:].*mk[:].*uk;
	  
	  A = GetHelmholtzOperator(M,mk[:],w,gamma[:],true,false);
	  V  = getVolume(M)
	  
  	  ut = A\(-rk);

  	  res = norm(A*ut + rk,Inf);

      err[k,1] = abs(V[1,1]*dot((ut.-uk),A*(ut.-uk)))
      err[k,2] = norm(ut-uk,Inf)
	
	  # figure()
	  # imshow(reshape(mk,M.n[1]+1,M.n[2]+1))
	  # imshow(gamma'); colorbar()
	  # subplot(2,2,1)
	  # imshow(reshape(real(ut),M.n[1]+1,M.n[2]+1));;colorbar()
	  # title("ut")
	  # subplot(2,2,2)
	  # imshow(reshape(real(uk),M.n[1]+1,M.n[2]+1));;colorbar()
	  # title("uk")
	  
	  # subplot(2,2,3)
	  # imshow(reshape(abs(uk-ut),M.n[1]+1,M.n[2]+1));colorbar()
	  # title("error")
	  
	  
      @printf "k=%d, n=[%d,%d], l2_err=%1.3e, factor=%1.3f linf_err=%1.3e\n" k M.n[1] M.n[2] err[k,1] err[max(k-1,1),1]/err[k,1] err[k,2]
   end

  # @test countnz(diff(log(err[:,1])).<-1.8) > 4
end

# Constant slowness
f = 1e-0;
w = 2*pi*f;
slowsq(x,y) = ones(size(x));
fictitiousSourceTest2D((x,y)->cos.(pi*x).*cos.(pi*y), slowsq,w,
      (x,y) -> (- (pi.^2).*cos.(pi*x).*cos.(pi*y) .- (pi.^2).*cos.(pi*x).*cos.(pi*y) .+ slowsq(x,y).*cos.(pi*x).*cos.(pi*y)*(w.^2)  ))

# # VARYING SLOWNESS TEST
slowsq(x,y) = exp.(-2.0*(x.^2 .+ y.^2));
fictitiousSourceTest2D((x,y)->cos.(pi*x).*cos.(pi*y), slowsq,w,
      (x,y) -> (- (pi.^2).*cos.(pi*x).*cos.(pi*y) .- (pi.^2).*cos.(pi*x).*cos.(pi*y) .+ slowsq(x,y).*cos.(pi*x).*cos.(pi*y)*(w.^2)  ))

println("\t== passed ! ==")
