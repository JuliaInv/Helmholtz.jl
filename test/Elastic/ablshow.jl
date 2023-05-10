
using Helmholtz
using PyPlot
using jInvVisPyPlot
close("all")
gamma = getABL([128,64],false,[8,8],1.0,0,[1 0 ; 1 1]);
figure();plotModel(gamma)

gamma = getABL([128,64],false,[8,8],1.0);
figure(); plotModel(gamma)


# gamma = getABL([128,128,64],false,[8,8,8],1.0);
# figure();plotModel(gamma)
