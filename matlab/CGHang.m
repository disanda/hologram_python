%% Demo of a CGH (computer-generated hologram)

clear;
%img0=double(rgb2gray(imread('pic\p1.bmp')));
img0=double(imread('p2.bmp'));
%img0=(rot90(img0));
img0=img0/max(max(abs(img0)));   %��һ��
img0=imresize(img0,[512 512]);
img0=enlarge_anysize(img0,1080,1080);           %zero padding ȫϢͼ�ߴ�Ϊ1080x1080����
imwrite(img0,'original.bmp','bmp'); %original object image ԭ����ͼ�񣬼ٶ���ά
%img0=zeros(hrow,hcol);

lamda=650e-9;                   %��������λ:��
z=0.2;                           %�����ȫϢͼ֮����룬��λ:��
dx=7.56e-6;  dy=dx;                 %����ͼ���ȫϢͼÿ�����ش�С����λ:��
[mm,nn]=size(img0);

[ du, obj ] = angular_spectrum(dx,lamda,img0,z);           
%ʹ�ý������䷽��ģ��Fresnel transform��objΪ���ɵĸ����ȫϢͼ
%ȫϢͼ������ǿ�Ⱥ���λ����ͼ�ֱ��ʾ��Ҳ������ʵ�����鲿����ͼ�ֱ��ʾ

intensity=abs(obj);
vmax=max(max(intensity));
intensitynorm=intensity/vmax;
imwrite(intensitynorm,'holointensity.bmp','bmp'); %ȫϢͼǿ�ȷֲ�

phase=angle(obj);
phasenorm=(phase+pi)/(2*pi);
imwrite(phasenorm,'holophase.bmp','bmp'); %ȫϢͼ��λ�ֲ�

realpart=real(obj);
vmax=max(max(realpart));
vmin=min(min(realpart));
realnorm=(realpart-vmin)/(vmax-vmin);
imwrite(realnorm,'holoreal.bmp','bmp'); %ȫϢͼʵ��

imagpart=imag(obj);
vmax=max(max(imagpart));
vmin=min(min(imagpart));
imagnorm=(imagpart-vmin)/(vmax-vmin);
imwrite(imagnorm,'holoimag.bmp','bmp');%ȫϢͼ�鲿


% �����ؽ�����ȫϢͼ���ؽ�������ͼ��
[ dv, img ] = angular_spectrum(du,lamda,obj,-z);             %���������ؽ�
fmag=abs(img);
vmin=min(min(fmag));
vmax=max(max(fmag));
vnorm=(fmag-vmin)/(vmax-vmin);
imwrite(vnorm,'rec.bmp','bmp');%�ؽ���������ͼ��

