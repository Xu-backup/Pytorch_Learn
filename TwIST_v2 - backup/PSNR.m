function PSNR = PSNR(f1, f2)

%kΪͼ���Ǳ�ʾ�ظ����ص����õĶ�����λ������λ�
B=8;            %����һ�������ö��ٶ�����λ
[h, w]=size(f1);
MAX=2^B-1;          %ͼ���ж��ٻҶȼ�
MES=(sum(sum(f1-f2).^2))/(h*w);     %������
PSNR=20*log10(MAX/sqrt(MES));           %��ֵ�����
