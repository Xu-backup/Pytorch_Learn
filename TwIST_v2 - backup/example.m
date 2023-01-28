y
k = sum(sum(f))

for i=1:8
    temp = input(:,:,i);
    figure(i);
    set(i,'Position',[512 512 300 300]);
    imagesc(temp);
    title("pic_"+i);
    drawnow;
end