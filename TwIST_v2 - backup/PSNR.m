function PSNR = PSNR(f1, f2)

%k为图像是表示地个像素点所用的二进制位数，即位深。
B=8;            %编码一个像素用多少二进制位
[h, w]=size(f1);
MAX=2^B-1;          %图像有多少灰度级
MES=(sum(sum(f1-f2).^2))/(h*w);     %均方差
PSNR=20*log10(MAX/sqrt(MES));           %峰值信噪比
