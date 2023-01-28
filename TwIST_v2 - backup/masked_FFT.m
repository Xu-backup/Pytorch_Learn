function y = masked_FFT(x,mask)
%
% Computes a 2D partial FFT transform,
% only for those frequencies for which 
% mask is not zero. The result is returned
% in an array of size (k,1) where k is
% the number of non-zeros in mask.
% The transpose of this operator is 
% available in the function masked_FFT_t.m
%
% Copyright, 2006, Mario Figueiredo.
% 
%
%[n1 n2] = size(x);

%ii = find(abs(mask)>0);
%Rf = fftshift(fft2(x)).*mask/n; 
%fft2二维快速傅里叶变换，输入与输出相同
%fftshift函数用于将变换后的图象频谱中心从矩阵的原点移到矩阵的中心
%y = Rf(ii);
inter = 1;

y = zeros(128+inter*7,128);

for i=1:8
    x(:,:,i)=x(:,:,i).*mask;
    y(1+(i-1)*inter:(i-1)*inter+128,1:128) = y(1+(i-1)*inter:(i-1)*inter+128,1:128) + x(:,:,i);
end
ymax = max(max(y));
ymin = min(min(y));
if(ymax-ymin~=0)
    y =(y-ymin)/(ymax-ymin);
end


%y=zeros(n);
%y(ii) = x(ii);


