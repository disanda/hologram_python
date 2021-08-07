%% Demo of a CGH (computer-generated hologram)

clear;
%img0=double(rgb2gray(imread('pic\p1.bmp')));
img0=double(imread('p2.bmp'));
%img0=(rot90(img0));
img0=img0/max(max(abs(img0)));   %归一化
img0=imresize(img0,[512 512]);
img0=enlarge_anysize(img0,1080,1080);           %zero padding 全息图尺寸为1080x1080像素
imwrite(img0,'original.bmp','bmp'); %original object image 原物体图像，假定二维
%img0=zeros(hrow,hcol);

lamda=650e-9;                   %波长，单位:米
z=0.2;                           %物体和全息图之间距离，单位:米
dx=7.56e-6;  dy=dx;                 %物体图像和全息图每个像素大小，单位:米
[mm,nn]=size(img0);

[ du, obj ] = angular_spectrum(dx,lamda,img0,z);           
%使用角谱衍射方法模拟Fresnel transform，obj为生成的复振幅全息图
%全息图可以用强度和相位两张图分别表示，也可以用实部和虚部两张图分别表示

intensity=abs(obj);
vmax=max(max(intensity));
intensitynorm=intensity/vmax;
imwrite(intensitynorm,'holointensity.bmp','bmp'); %全息图强度分布

phase=angle(obj);
phasenorm=(phase+pi)/(2*pi);
imwrite(phasenorm,'holophase.bmp','bmp'); %全息图相位分布

realpart=real(obj);
vmax=max(max(realpart));
vmin=min(min(realpart));
realnorm=(realpart-vmin)/(vmax-vmin);
imwrite(realnorm,'holoreal.bmp','bmp'); %全息图实部

imagpart=imag(obj);
vmax=max(max(imagpart));
vmin=min(min(imagpart));
imagnorm=(imagpart-vmin)/(vmax-vmin);
imwrite(imagnorm,'holoimag.bmp','bmp');%全息图虚部


% 仿真重建：从全息图中重建出物体图像
[ dv, img ] = angular_spectrum(du,lamda,obj,-z);             %角谱衍射重建
fmag=abs(img);
vmin=min(min(fmag));
vmax=max(max(fmag));
vnorm=(fmag-vmin)/(vmax-vmin);
imwrite(vnorm,'rec.bmp','bmp');%重建出的物体图像

