load("D:\SourceCode\PyTorch\DeepCUP\result\result.mat");
s = 0;
s2 = 0;
for k=1:8
    a = output(k,:,:);
    a = squeeze(a);
    b = target(:,:,k);
    temp = psnr(a,b);   
    temp2 = ssim(a,b);
    s = s + temp;
    s2 = s2 + temp2;
end
avg = s/8
avg2 = s2/8

s = 0;
s2 = 0;
