clearvars;
%x12 is 2nd column of 1st class.
[x11_train,x12_train]=textread('..\data_assign3_group5\linearly_separable\class1_train.txt','%f %f');
[x11_test,x12_test]=textread('..\data_assign3_group5\linearly_separable\class1_test.txt','%f %f');
[x11_val,x12_val]=textread('..\data_assign3_group5\linearly_separable\class1_val.txt','%f %f');

[x21_train,x22_train]=textread('..\data_assign3_group5\linearly_separable\class2_train.txt','%f %f');
[x21_test,x22_test]=textread('..\data_assign3_group5\linearly_separable\class2_test.txt','%f %f');
[x21_val,x22_val]=textread('..\data_assign3_group5\linearly_separable\class2_val.txt','%f %f');

[x31_train,x32_train]=textread('..\data_assign3_group5\linearly_separable\class3_train.txt','%f %f');
[x31_test,x32_test]=textread('..\data_assign3_group5\linearly_separable\class3_test.txt','%f %f');
[x31_val,x32_val]=textread('..\data_assign3_group5\linearly_separable\class3_val.txt','%f %f');

[x41_train,x42_train]=textread('..\data_assign3_group5\linearly_separable\class4_train.txt','%f %f');
[x41_test,x42_test]=textread('..\data_assign3_group5\linearly_separable\class4_test.txt','%f %f');
[x41_val,x42_val]=textread('..\data_assign3_group5\linearly_separable\class4_val.txt','%f %f');

[Ntrain1,nq]=size(x11_train);
[Ntest1,nq]=size(x11_test);
[Nval1,nq]=size(x11_val);

[Ntrain2,nq]=size(x21_train);
[Ntest2,nq]=size(x21_test);
[Nval2,nq]=size(x21_val);

[Ntrain3,nq]=size(x31_train);
[Ntest3,nq]=size(x31_test);
[Nval3,nq]=size(x31_val);

[Ntrain4,nq]=size(x41_train);
[Ntest4,nq]=size(x41_test);
[Nval4,nq]=size(x41_val);

 N=Ntrain1+Ntrain2+Ntrain3+Ntrain4;

 X1_train=[x11_train,x12_train];
 X2_train=[x21_train,x22_train];
 X3_train=[x31_train,x32_train];
 X4_train=[x41_train,x42_train];

 X1_test=[x11_test,x12_test];
 X2_test=[x21_test,x22_test];
 X3_test=[x31_test,x32_test];
 X4_test=[x41_test,x42_test];
 
 Y1_train=zeros(Ntrain1,1);
 Y2_train=zeros(Ntrain2,1);
 Y3_train=zeros(Ntrain3,1);
 Y4_train=zeros(Ntrain4,1);
 
 for i=1:Ntrain1 Y1_train(i)=1; end
 for i=1:Ntrain2 Y2_train(i)=2; end
 for i=1:Ntrain3 Y3_train(i)=3; end
 for i=1:Ntrain4 Y4_train(i)=4; end
 
 for i=1:Ntest1 Y1_test(i)=1; end
 for i=1:Ntest2 Y2_test(i)=2; end
 for i=1:Ntest3 Y3_test(i)=3; end
 for i=1:Ntest4 Y4_test(i)=4; end

a = [0 1]';
b = [1 0]';
c = [1 1]';
d = [0 0]';

hold on
scatter(x11_train,x12_train,'g');
scatter(x21_train,x22_train,'b');
scatter(x31_train,x32_train,'m');
scatter(x41_train,x42_train,'r');


 x=[X1_train' X2_train' X3_train' X4_train'];
 T = [repmat(a,1,length(X1_train)) repmat(b,1,length(X2_train)) ...
      repmat(c,1,length(X3_train)) repmat(d,1,length(X4_train))];

predictions=zeros(2,1000);

net=perceptron;
E = 1;
net.adaptParam.passes = 1;
linehandle = plotpc(net.IW{1},net.b{1});
n = 0;
while (sse(E) && n<1000)
   n = n+1;
   [net,F,E] = adapt(net,x,T);
   linehandle = plotpc(net.IW{1},net.b{1},linehandle);
   drawnow;
   disp(n);
end
%show perceptron structure

predicted1=zeros(Ntest1,1);

xrange = [-15 20];
yrange = [-15 20];
inc = 0.1;
[g, h] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(g);
xy = [g(:) h(:)];
xy = [reshape(g, image_size(1)*image_size(2),1) reshape(h, image_size(1)*image_size(2),1)];
Ntest=length(xy);

predicted5=zeros(Ntest,4);
predicted=zeros(Ntest,1);

for i=1:Ntest
  xx=xy(i,:);
  p = [xx(1);xx(2)];
  tp=net(p)';
  tp1=[tp(1);tp(2)];
%  disp(i);
 if(isequal(a,tp1)) 
     predicted(i)=1;
 elseif(isequal(b,tp1))
     predicted(i)=2;
 elseif(isequal(c,tp1))
     predicted(i)=3;
 else
     predicted(i)=4;
 end
end

decisionmap = reshape(predicted, image_size);

figure;
imagesc(xrange,yrange,decisionmap);

hold on

set(gca,'ydir','normal');
cmap = [0.7 1 0.7;0.9 0.9 1;1 0.7 1;1 0.8 0.8];
colormap(cmap);

%scatter(x11_train,x12_train,'g');
%scatter(x21_train,x22_train,'b');
%scatter(x31_train,x32_train,'m');
%scatter(x41_train,x42_train,'r');
%hold off
%  
% for i=1:Ntrain1 
%  p = [x11_train(i);x12_train(i)];
%  tp=net(p)';
%  tp1=[tp(1);tp(2)];
%  
%  if(isequal(a,tp1)) 
%      predicted1(i)=1;
%  elseif(isequal(b,tp1))
%      predicted1(i)=2;
%  elseif(isequal(c,tp1))
%      predicted1(i)=3;
%  else
%      predicted1(i)=4;
%  end
% end
%view(net);
