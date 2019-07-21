function [gFilter]= gaussianFilter(width, power, rate)
X=linspace(-1, 1, width);
[X,Y]=meshgrid(X,X);
gFilter=abs(X).^power + abs(Y).^power;
% disp(([gFilter(1, width/2), gFilter(width/2, 1),...
%     gFilter(width/2, width), gFilter(width, width/2)]))

gFilter=exp(-gFilter.*rate);
% nnz(gFilter>0.1)

end