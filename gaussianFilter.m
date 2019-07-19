function [gFilter]= gaussianFilter(width, power, rate)
gFilter = zeros(width,width);
x0=width/2;y0=width/2;
for col = 1 : size(gFilter,1)
  for row = 1 : size(gFilter,2)
    gFilter(row, col) =...
        exp(-((abs(col-x0)/x0)^power+(abs(row-y0)/y0)^power)*rate);
  end
end

end