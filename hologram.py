# angular_spectrum for hologram
import cv2
import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2,ifftshift,fftshift

image_name = 'p2.bmp'

def angular_spectrum(dx, r, img, z):
    #ANGULAR_SPECTRUM Summary of this function and Detailed explanation goes here
    # r=r0/n，及波长是光在介质中的波长，而非真空中的波长.
    # 角谱传播函数，正向传播z为正，反向传播z为负'.
    # strcat('z = ',num2str(z)))

    dy = dx;
    du = dx;
    nn,mm = img.shape
    dfx = 1/(dx*nn);
    dfy = 1/(dy*mm);

    pha = np.zeros((nn,mm))
    for ii in range(0,nn):
        for jj in range(0,mm):
            #pha(ii,jj) = dfx^2*(ii-nn/2-0.5)^2 + dfy^2*(jj-mm/2-0.5)^2
            pha[ii,jj] = dfx**2*(ii-nn/2-0.5)**2 + dfy**2*(jj-mm/2-0.5)**2 # fx^2 + fy^2

    #e_pha = exp(1i*2*pi*z/r.*sqrt(1-r^2.*pha)) # from matlab
    e_pha = np.exp( (1j * 2 * np.pi * z/r ) * np.sqrt(1-r**2 * pha)) 

    tmp = fftshift(fft2(fftshift(img)))
    tmp = tmp*e_pha
    img = fftshift(ifft2(fftshift(tmp)))
    return img

def zero_padding(img_grey,h_,w_): # from cv2 [c,h,w]
#本函数将输入的灰度图img_grey四周加黑框(zero_padding)，得到h_*w_分辨率的灰度图img_out；
    h, w = img_grey.shape
    t1 = int(np.ceil( (h_-h) /2 ))
    t2 = int(np.fix(  (h_-h) /2 ))
    s1 = int(np.ceil( (w_-w) /2 ))
    s2 = int(np.fix(  (w_-w) /2 ))

    T1 = np.zeros((h,s1))     #左右两边各扩展s列
    T2 = np.zeros((h,s2))
    T3 = np.zeros((t1,w+s1+s2)) #上下两端各扩展t行
    T4 = np.zeros((t2,w+s1+s2))

    #MatLab: IMGout=[T1,IMGin,T2]; IMGout=[T3',IMGout',T4']'; X' = X.T (矩阵转置)
    # print(T1.shape)
    # print(T2.shape)
    img_out = np.c_[T1,img_grey] #列拼接
    img_out = np.c_[img_out,T2]
    img_out = np.r_[T3,img_out] #行拼接
    img_out = np.r_[img_out,T4]
    return img_out

img = cv2.imread(image_name)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img/255.0 # [0,255] -> [0,1]
img = zero_padding(img,1080,1080)
img = img/np.max(np.max(np.abs(img)))

image_name = image_name.rstrip('.bmp') #删除指定字符
image_name = image_name.rstrip('.jpg')
image_name = image_name.rstrip('.png')
cv2.imwrite('origin_%s.bmp'%image_name, img*255.0)

lamda=650e-9 # 波长，单位:米
z=0.2 # 物体和全息图之间距离，单位:米
dx=7.56e-6
dy=dx # 物体图像和全息图每个像素大小，单位:米
mm,nn = img.shape

img_as = angular_spectrum(dx,lamda,img,z)

intensity=np.abs(img_as);
vmax=np.max(np.max(intensity));
intensitynorm=intensity/vmax*255.0;
cv2.imwrite('./holointensity_%s.bmp'%image_name,intensitynorm) #全息图强度分布

phase=np.angle(img_as);
phasenorm=(phase+np.pi)/(2*np.pi)*255.0;
cv2.imwrite('holophase_%s.bmp'%image_name, phasenorm) #全息图相位分布

realpart=np.real(img_as);
vmax=np.max(np.max(realpart));
vmin=np.min(np.min(realpart));
realnorm=(realpart-vmin)/(vmax-vmin)*255.0;
cv2.imwrite('holoreal_%s.bmp'%image_name, realnorm) #全息图实部

imagpart=np.imag(img_as);
vmax=np.max(np.max(imagpart));
vmin=np.min(np.min(imagpart));
imagnorm=(imagpart-vmin)/(vmax-vmin)*255.0;
cv2.imwrite('holoimag_%s.bmp'%image_name, imagnorm) #全息图虚部

#仿真重建：从全息图中重建出物体图像
img = angular_spectrum(dx,lamda,img_as,-z) #角谱衍射重建
fmag=np.abs(img);
vmin=np.min(np.min(fmag));
vmax=np.max(np.max(fmag));
vnorm=(fmag-vmin)/(vmax-vmin)*255.0;
cv2.imwrite('rec_%s.bmp'%image_name, vnorm) #重建出的物体图像
