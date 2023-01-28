function y = masked_FFT_t(x,mask)
%
% This is the transpose operator of the
% partial FFT transform, implemented in 
% masked_FFT.m.  See that file for details.
%
y = zeros(128,128,8);
% ii=find(abs(mask)>0);
% gg = zeros(n,n); 
% gg(ii)=x(ii);
% %y = real(ifft2(ifftshift(gg)))*n;
% y = gg;
y = make_8(x,1);



