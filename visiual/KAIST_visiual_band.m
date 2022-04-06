addpath('./packages');
%s1 = result(1:28,:,:);
%s = permute(s1,[2,3,1]);
%s = reshape(output,256,256,28);
%s = label;
%s= res;
s = recon(7,:,:,:);
s = reshape(s,256,256,28);
%s = vgaptv;
%s1 = result(169:196,:,:);
%s = permute(s1,[2,3,1]);
band = [453.3, 457.6, 462.1, 466.8, 471.6, 476.5,...
    481.6, 486.9, 492.4, 498.0, 503.9, 509.9, 516.2, 522.7, 529.5, 536.5, 543.8, 551.4,...
    558.6, 567.5, 575.3, 584.3, 594.4, 604.2, 614.4, 625.1, 636.3, 648.1];
%s = label;
%s = res;
%s= vgaptv;
%s=vdesci;

for i =1:28
    vshowSpectralData(s(:,:,i),band(i),'magsize');
    set(gcf,'Position',[0,0,256,256]);
    saveas(gcf,['.\scene7\tsa_band\',num2str(i)],'png');
end

