clear all
T = [0.005 0.007 0.012 0.015 0.023 0.030 0.026 0.024 0.019 0.010 0.004 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000;...
     0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.020 0.013 0.011 0.009 0.005 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003;...
     0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012 0.013 0.022 0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];
ST=sum(T,2);

size_input = 48;
hyperspectralSlices=1:31;

load_imgDataPath = '.\'


addpath(load_imgDataPath);


imgdir = dir(load_imgDataPath);
psnr_ssim_sam = zeros(length(imgdir)-2,3,'single');
stride = 24;


for i = 3:length(imgdir)
	if (isequal(imgdir(i).name,'.')||isequal(imgdir.name,'..'))
		continue;
	end
	load([load_imgDataPath,'\',imgdir(i).name]);;%load test_out�?output and label
	imgname = imgdir(i).name(1:end-4);%去掉.mat
	output = output.*(output>0);
	[h,w,c]=size(label);
	ture_img = label;
    
	result_img = output(1,:,:,:);
    result_img = reshape(result_img,h,w,c); 
	%psnr
	maxi = max(max(ture_img));
	mse = mean(mean((result_img-ture_img).^2));
	psnr = mean(10*log10(1.^2./mse));
	%ssim
	c1=0.01;
	c2=0.03;
	ssim_tmp=zeros(1,28);
	for k = 1:28
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
	
	disp([num2str(i-2), '-', num2str(psnr), '-', num2str(ssim), '-', num2str(sam)]);
    Result_psnr_ssim(i-2,:)=[psnr,ssim,sam];
	
end
Result_psnr_ssim
meanResult=mean(Result_psnr_ssim)