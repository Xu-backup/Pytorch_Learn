function demo_MRI
% MRI demo: Reconstruction from noisy random projections
%
% This demo illustrates the twIST algorithm computing the solution  of  
%
%     xe = arg min 0.5*||A x-y||^2 + tau TV(x),
%             x
%  where x is an image, Ax is the Fourier Transform of x  computed in a 
%  "small" set of frequencies, y is a noisy observation and TV(x) is
%  the total variation of x.
% 
% The proximal operator, i.e., the solution of
%
%     xe = arg min 0.5 ||x-y||^2 + tau TV(x)
%             x
% is  given by the Chambolle algorithm
% 
% A. Chambolle, 揂n algorithm for total variation minimization and
% applications,? Journal of Mathematical Imaging and Vision, vol. 20,
% pp. 89-97, 2004.
%
%
% For further details about the TwIST algorithm, see the paper:
%
% J. Bioucas-Dias and M. Figueiredo, "A New TwIST: Two-Step
% Iterative Shrinkage/Thresholding Algorithms for Image 
% Restoration",  IEEE Transactions on Image processing, 2007.
%
%
% (available at   http://www.lx.it.pt/~bioucas/publications.html)
%
%
% Authors: Jose Bioucas-Dias and Mario Figueiredo, 
% Instituto Superior T閏nico, October, 2007
%
%CT图重构
% generate phantom
%f = phantom(256);  %生成头部CT图
%f =  f - sum(sum(f))/prod(size(f));

f = double(imread('cameraman.tif'));
mu=mean(f(:));
f=f-mu;


n =length(f);
angles = 22;
fprintf(1,'Generating beam mask with %d angles\n',angles)
%mask = MRImask(n,angles);
mask = rand(n)>(1-0.5); %nxn的随机矩阵 大于0.8的置为1

fprintf(1,'Done\n');
k = sum(sum(mask)); %sum计算每一列和，两个sum计算所有


fprintf(1, ' Problem Dimensions: n=%d, k=%d\n', n, k);

% Measurement noise standard deviation
sigma = 1e-3; %噪声水平


% create handle for observation operator and transpose
hR = @(x)  masked_FFT(x,mask); %快速傅里叶变换
hRT = @(x) masked_FFT_t(x,mask); %逆过程

% vector of observations
y_noiseless = hR(f);
y = y_noiseless + sigma*randn(size(y_noiseless));
z = hRT(y)


figure(10)
set(10,'Position',[10 40 300 300])
imagesc(f)
colormap(gray)
axis image
axis off
title('Original')
drawnow

figure(11)
set(11,'Position',[10 400 300 300])
imagesc(y)
colormap(gray)
axis image
axis off
title('Back-projection')
drawnow

figure(9)
set(9,'Position',[10 400 300 300])
imagesc(z)
colormap(gray)
axis image
axis off
title('Back-projection-1')
drawnow



% denoising function;
tv_iters = 10;
Psi = @(x,th)  tvdenoise(x,2/th,tv_iters);

% set the penalty function, to compute the objective
Phi = @(x) TVnorm(x);

% regularization parameters (empirical)
tau = max(norm(y,'fro')/sqrt(4*k)/4,sqrt(3)*sigma);
tau = 0.001;

tolA = 1e-4;
% -- TwIST ---------------------------
% stop criterium:  the relative change in the objective function 
% falls below 'ToleranceA'         
 [x_twist,dummy,obj_twist,...
    times_twist,dummy,mse_twist]= ...
         TwIST(y,hR,tau,...
         'Lambda', 1e-3, ...
         'AT', hRT, ...
         'Psi', Psi, ...
         'Phi',Phi, ...
         'True_x', f, ...
         'Monotone',1,...
         'MaxiterA', 10000, ...
         'Initialization',0,...
         'StopCriterion',1,...
       	 'ToleranceA',tolA,...
         'Verbose', 1);
   
% [x_twist,dummy,obj_twist,...
%     times_twist,dummy,mse_twist]= ...
%          SpaRSA(y,hR,tau,...
%          'AT', hRT, ...
%          'Psi', Psi, ...
%          'Phi',Phi, ...
%          'True_x', f, ...
%          'BB_factor', 0.8, ...
%          'Monotone',1,...
%          'Initialization',0,...
%          'StopCriterion',4,...
%        	 'ToleranceA',obj_twist(end),...
%          'Verbose', 1);


figure(12);
set(12,'Position',[400 385 300 300])
imagesc(x_twist);
axis image
title('Estimate')
colormap(gray)

psnr = PSNR(x_twist,f)

figure(14);
set(14,'Position',[800 0 300 300])
semilogy(times_twist, (mse_twist*prod(size(f))).^0.5/norm(f,'fro'),'LineWidth',2);
title('error ||x^{t}-x||_2/||x||_2')
xlabel('CPU time (sec)')
grid on

figure(13)
set(13,'Position',[800 400 300 300])
semilogy(times_twist,obj_twist,'b','LineWidth',2)
title('Objective function')
xlabel('CPU time (sec)')
grid on

