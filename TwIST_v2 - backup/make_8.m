function output = make_8(input,inter)
%MAKE_8 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

output = zeros(128,128,8);
for i=1:8
    output(:,:,i) = input(1+(i-1)*inter:128+(i-1)*inter,1:128);
end
end

