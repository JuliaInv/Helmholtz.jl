function plotVectorU2D(u::Array,Minv::RegularMesh,fig_title::String;clim = [minimum(real(u)),maximum(real(u))])
	dim = Minv.dim;
	ux = getUjProjMatrix(Minv.n,1)'*u;
	uz = getUjProjMatrix(Minv.n,dim)'*u;

	ux = reshape(ux,(Minv.n[1]+1,Minv.n[dim]));
	uz = reshape(uz,(Minv.n[1],Minv.n[dim]+1));
	
	subplot(1,2,1)
	imshow(real(ux)',clim = clim);colorbar()
	title(string(fig_title,"_x"))
	subplot(1,2,2)
	imshow(real(uz)',clim = clim);colorbar()
	title(string(fig_title,"_z"))
end


function plotVectorU2D(u::Array,us::Array,Minv::RegularMesh,fig_title::Tuple{String,String};clim = [min(minimum(u),minimum(us)),max(maximum(u),maximum(us))], cmap = "jet")
	ux = getUjProjMatrix(Minv.n,1)'*u;
	uy = getUjProjMatrix(Minv.n,2)'*u;

	ux = reshape(ux,(Minv.n[1]+1,Minv.n[2]));
	uy = reshape(uy,(Minv.n[1],Minv.n[2]+1));


	usx = getUjProjMatrix(Minv.n,1)'*us;
	usy = getUjProjMatrix(Minv.n,2)'*us;

	usx = reshape(usx,(Minv.n[1]+1,Minv.n[2]));
	usy = reshape(usy,(Minv.n[1],Minv.n[2]+1));

	subplot(2,2,1)
	imshow(real(ux)',clim = clim,cmap = cmap);colorbar()
	title(string(fig_title[1],"_x"))
	subplot(2,2,2)
	imshow(real(uy)',clim = clim,cmap = cmap);colorbar()
	title(string(fig_title[1],"_z"))
	subplot(2,2,3)
	imshow(real(usx)',clim = clim,cmap = cmap);colorbar()
	title(string(fig_title[2],"_x"))
	subplot(2,2,4)
	imshow(real(usy)',clim = clim,cmap = cmap);colorbar()
	title(string(fig_title[2],"_z"))
end