
function loc2cs(loc1::Int64,loc2::Int64,n::Array{Int64,1})
@inbounds cs = loc1 + (loc2-1)*n[1];
return cs;
end

function loc2cs3D(loc1::Int64,loc2::Int64,loc3::Int64,n::Array{Int64,1})
@inbounds cs = loc1 + (loc2-1)*n[1] + (loc3-1)*n[1]*n[2];
return cs;
end



function My_sub2ind(n::Array{Int64},sub::Array{Int64})
if length(sub)==2
	return loc2cs(sub[1],sub[2],n);
else
	return loc2cs3D(sub[1],sub[2],sub[3],n);
end
end