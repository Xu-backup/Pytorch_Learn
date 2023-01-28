function y = TVnorm(x)
y = 0;
for i=1:8
    temp = x(:,:,i); 
    y = y + sum(sum(sqrt(diffh(temp).^2+diffv(temp).^2)));
end
%y = y/8;