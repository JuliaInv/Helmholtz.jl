export getAcousticPointSource,loc2cs,getElasticPointSource,getElasticPointSourceMid,getMidPointSrc;


function getElasticPointSource(Minv,TYPE)
## Generating the right hand side
if Minv.dim==3
	qx = zeros(TYPE,tuple(Minv.n + [1; 0; 0]...));
	qy = zeros(TYPE,tuple(Minv.n + [0; 1; 0]...));
	qz = zeros(TYPE,tuple(Minv.n + [0; 0; 1]...));
	src = [div(Minv.n[1],2),div(Minv.n[2],2),1];
	qz[src[1],src[2],1] = -2.0./(Minv.h[1]);
	qx[src[1],src[2],1] = -1.0./(Minv.h[3]);
	qx[src[1]+1,src[2],1] = +1.0./(Minv.h[3]);
	qy[src[1],src[2],1] = -1.0./(Minv.h[3]);
	qy[src[1],src[2]+1,1] = +1.0./(Minv.h[3]);
	q = vec([qx[:]; qy[:]; qz[:];]);
else
	qx = zeros(TYPE,tuple(Minv.n + [1; 0]...))
	qy = zeros(TYPE,tuple(Minv.n + [0; 1]...))
	qy[div(Minv.n[1],2),1] = -2.0./(Minv.h[1]);
	#qy[div(Minv.n[1],2),2] = -1.0./(Minv.h[1]);
	qx[div(Minv.n[1],2),1] = -1.0./(Minv.h[2]);
	qx[div(Minv.n[1],2)+1,1] = +1.0./(Minv.h[2]);
	# qy[div(Minv.n[1],2),div(Minv.n[2],2)] = -2./(Minv.h[1]);
	# qx[div(Minv.n[1],2),div(Minv.n[2],2)] = -1./(Minv.h[2]);
	# qx[div(Minv.n[1],2)+1,div(Minv.n[2],2)] = +1./(Minv.h[2]);	
	q = vec([qx[:]; qy[:]])
end
return q;
end


function getElasticPointSourceMid(Minv,TYPE)
## Generating the right hand side
if Minv.dim==3
	qx = zeros(TYPE,tuple(Minv.n + [1; 0; 0]...));
	qy = zeros(TYPE,tuple(Minv.n + [0; 1; 0]...));
	qz = zeros(TYPE,tuple(Minv.n + [0; 0; 1]...));
	error("getElasticPointSourceMid: Correct below:")
	src = [div(Minv.n[1],2),div(Minv.n[2],2),1];
	qz[src[1],src[2],1] = -2.0./(Minv.h[1]);
	qx[src[1],src[2],1] = -1.0./(Minv.h[3]);
	qx[src[1]+1,src[2],1] = +1.0./(Minv.h[3]);
	qy[src[1],src[2],1] = -1.0./(Minv.h[3]);
	qy[src[1],src[2]+1,1] = +1.0./(Minv.h[3]);
	q = vec([qx[:]; qy[:]; qz[:];]);
else
	qx = zeros(TYPE,tuple(Minv.n + [1; 0]...))
	qy = zeros(TYPE,tuple(Minv.n + [0; 1]...))
	n_mid = div.(Minv.n,2);
	qy[n_mid[1],n_mid[2]] = -1.0./(Minv.h[1]^2);
	qy[n_mid[1],n_mid[2]+1] = 1.0./(Minv.h[1]^2);
	qx[n_mid[1],n_mid[2]] = -1.0./(Minv.h[2]^2);
	qx[n_mid[1]+1,n_mid[2]] = 1.0./(Minv.h[2]^2);
	q = vec([qx[:]; qy[:]])
end
return q;
end




function getTopPointSrc(Minv)
if Minv.dim==3
	src = [div(Minv.n[1]+1,2);div(Minv.n[2]+1,2);1];
else
    src = [div(Minv.n[1]+1,2);1]
end
return src;
end

function getMidPointSrc(Minv)
if Minv.dim==3
	src = [div(Minv.n[1]+1,2);div(Minv.n[2]+1,2);div(Minv.n[3]+1,2)];
else
    src = [div(Minv.n[1]+1,2);div(Minv.n[2]+1,2)]
end
return src;
end


function loc2cs(loc1::Int64,loc2::Int64,n::Array{Int64,1})
@inbounds cs = loc1 + (loc2-1)*n[1];
return cs;
end

function loc2cs3D(loc1::Int64,loc2::Int64,loc3::Int64,n::Array{Int64,1})
@inbounds cs = loc1 + (loc2-1)*n[1] + (loc3-1)*n[1]*n[2];
return cs;
end



function loc2cs(n::Array{Int64},sub::Array{Int64})
if length(sub)==2
	return loc2cs(sub[1],sub[2],n);
else
	return loc2cs3D(sub[1],sub[2],sub[3],n);
end
end




function getAcousticPointSource(Minv,TYPE,src = getTopPointSrc(Minv))
## Generating the right hand side
n_nodes = Minv.n .+ 1;
q = zeros(TYPE,tuple(n_nodes...));
q[loc2cs(n_nodes,src)] = 1.0./(norm(Minv.h)^2);
return q,src;
end

