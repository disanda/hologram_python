function IMGout=enlarge_anysize(IMGin,mm,nn);
%本函数将输入的灰度图IMGin四周加黑框，得到512*512分辨率的灰度图IMGout；
%本函数不判断mn是否超出512，该判断由调用本函数的程序执行；

[m,n]=size(IMGin);   %m是行数，n是列数；

%此处添加若m、n为奇数情况的处理办法：先添加一行或一列

t1=ceil((mm-m)/2);    %单边需要扩展的行数
t2=fix((mm-m)/2);
s1=ceil((nn-n)/2);    %单边需要扩展的列数
s2=fix((nn-n)/2);

T1=zeros(m,s1);     %左右两边各扩展s列
T2=zeros(m,s2);
T3=zeros(t1,n+s1+s2); %上下两端各扩展t行
T4=zeros(t2,n+s1+s2);

IMGout=[T1,IMGin,T2];
IMGout=[T3',IMGout',T4']';

return

%//////////////////////////////////////////////////////

%     //////////
%     /F matrix/
%     //////////
%
%       =>
%
%    //////////////////////     \
%    //    (frg_gray)    //     /  t lines
%    //    //////////    //
%    //    /F matrix/    //
%    //    //////////    //
%    //                  //
%    //////////////////////
%    \    /
%   s columns