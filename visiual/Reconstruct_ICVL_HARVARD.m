clear all
T = [0.005 0.007 0.012 0.015 0.023 0.030 0.026 0.024 0.019 0.010 0.004 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000;...
     0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.020 0.013 0.011 0.009 0.005 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003;...
     0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012 0.013 0.022 0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];
ST=sum(T,2);

size_input = 48;
hyperspectralSlices=1:31;

load_imgDataPath = '.\'
savePath = '.\Result';

addpath(load_imgDataPath);
addpath(savePath);

imgdir = dir(load_imgDataPath);
psnr_ssim_sam = zeros(length(imgdir)-2,3,'single');
stride = 24;


for i = 3:length(imgdir)-1
	if (isequal(imgdir(i).name,'.')||isequal(imgdir.name,'..'))
		continue;
	end
	load([load_imgDataPath,'\',imgdir(i).name]);;%load test_out‰∏?output and label
	imgname = imgdir(i).name(1:end-4);%ÂéªÊéâ.mat
	output = output.*(output>0);
	[h,w,c]=size(label);
	count = 1;
	result_image = zeros(size(label));
	result_weight = zeros(size(label));
	%image reconstruct
	for x=1 : stride : h-size_input+1-31+1
		for y = 1: stride : w-size_input+1
			tmp_out = output(:,:,:,count);
			for z = 1:31
				result_image(x+z-1:x+size_input-1+z-1,y:y+size_input-1,z) = result_image(x+z-1:x+size_input-1+z-1,y:y+size_input-1,z)+tmp_out(:,:,z);
				result_weight(x+z-1:x+size_input-1+z-1,y:y+size_input-1,z) = result_weight(x+z-1:x+size_input-1+z-1,y:y+size_input-1,z)+1;
			end
			count = count +1;
		end
	end
	
	
	result_image = result_image(1+35:end-35,1+8:end-8,:);
	result_weight = result_weight(1+35:end-35,1+8:end-8,:);
	label = label(1+35:end-35,1+8:end-8,:);
	result = result_image./result_weight;
	result = result.*(result>0);
	
	
	
	%hyper2rgb
	rgb_recon=zeros(512,512,3);
	rgb_gt=zeros(512,512,3);
	x_recon = result;
    gt = label;
	%ÂÉèÁ¥†ÂÄºÂΩí‰∏?åñ
	for ch = 1:31
		x_recon(:,:,ch)=x_recon(:,:,ch)/(max(max(gt(:,:,ch))));
		gt(:,:,ch) = gt(:,:,ch)./max(max(gt(:,:,ch)));
	end
	
	for ch = 1:31
		rgb_recon(:,:,1)= rgb_recon(:,:,1)+T(1,ch)*x_recon(:,:,ch);
		rgb_recon(:,:,2) = rgb_recon(:,:,2)+T(2,ch)*x_recon(:,:,ch);
		rgb_recon(:,:,3) = rgb_recon(:,:,3)+T(3,ch)*x_recon(:,:,ch);
		
		rgb_gt(:,:,1) = rgb_gt(:,:,1)+T(1,ch)*gt(:,:,ch);
		rgb_gt(:,:,2)= rgb_gt(:,:,2)+T(2,ch)*gt(:,:,ch);
		rgb_gt(:,:,3)= rgb_gt(:,:,3)+T(3,ch)*gt(:,:,ch);
	end
    for ch=1:3
        tt=ST(ch);
        rgb_recon(:,:,ch)=rgb_recon(:,:,ch)./tt;
        rgb_gt(:,:,ch)=rgb_gt(:,:,ch)./tt;
    end
	
	tmp = rgb_gt(:,:,1);
	rgb_gt(:,:,1)=rgb_gt(:,:,3);
	rgb_gt(:,:,3)=tmp;
	
	tmp = rgb_recon(:,:,1);
	rgb_recon(:,:,1)=rgb_recon(:,:,3);
	rgb_recon(:,:,3)=tmp;
	
    filleName=fullfile(savePath,'RGB',imgname);
    if ~exist(filleName,'file')
        mkdir(filleName);
    end
    imwrite(rgb_recon,[filleName,'\rgb_ours.png']);
    imwrite(rgb_gt,[filleName,'\rgb_gt.png']);
	
	ture_img = label;
	result_img = result;
	%psnr
	maxi = max(max(ture_img));
	mse = mean(mean((result_img-ture_img).^2));
	psnr = mean(10*log10(maxi.^2./mse));
	%ssim
	c1=0.01;
	c2=0.03;
	ssim_tmp=zeros(1,31);
	for k = 1:31
		a=2*mean2(ture_img(:,:,k))*mean2(result_img(:,:,k))+c1^2;
		cov_x=cov(ture_img(:,:,k),result_img(:,:,k));
		b=2*cov_x(1,2)+c2^2;
		c=mean2(ture_img(:,:,k))^2+mean2(result_img(:,:,k))^2+c1^2;
		d=cov_x(1,1)+cov_x(2,2)+c2^2;
		ssim_tmp(1,k)=a*b/c/d;
	end
	ssim = mean(ssim_tmp);
	%sam
	tmp= sum(ture_img.*result_img,3)./sqrt(sum(ture_img.^2,3))./sqrt(sum(result_img.^2,3));
	sam= mean2(real(acos(tmp)));
	
    %ergas
    ergas=0;
	for k = 1:31
        ergas = ergas + mean(mean((result_img(:, :, k)-ture_img(:, :, k)).^2))/((mean2(ture_img(:, :,k))).^2);
    end
    ergas = 100*sqrt(ergas/31);
    
	disp([num2str(i-2), '-', num2str(psnr), '-', num2str(ssim), '-', num2str(sam),'-', num2str(ergas)]);
    Result_psnr_ssim(i-2,:)=[psnr,ssim,sam,ergas];
	
end
Result_psnr_ssim
meanResult=mean(Result_psnr_ssim)