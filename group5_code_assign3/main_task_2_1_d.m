clearvars;
load('image_train_poly');
%load('image_poly_pca');
%load('partial_data');
%x12 is 2nd column of 1st class.

[size1,nq]=size(class1);
[size2,nq]=size(class2);
[size3,nq]=size(class3);

Ntrain1=size1;
Ntrain2=size2;
Ntrain3=size3;

Ntest1=84;
Ntest2=70;
Ntest3=58;

X1=class1;
X2=class2;
X3=class3;

 N=Ntrain1+Ntrain2+Ntrain3;

 reduced_d=2;
 
x_pca=vertcat(class1,class2,class3);
eigen=pca(x_pca);
transformed_vec=x_pca*eigen(:,[1,reduced_d]);


X1=zeros(13464,reduced_d);
X2=zeros(12960,reduced_d);
X3=zeros(10512,reduced_d);

 M=2;
 d=reduced_d;
 syms x y;
 
 D = factorial(M+d) / (factorial(d)*factorial(M)); 
 [X,Y] = meshgrid(0:M);
 XY = [X(:),Y(:)];
 
 pol=zeros(D,2);
 len = length(XY);
 j=1;
 for i =1 : len
     sum=XY(i,1)+XY(i,2);
     if(sum<= M)
        pol(j,1) = XY(i,1);
        pol(j,2) = XY(i,2);
        j = j +1;
     end
 end
expression = expand((x+y).^0);  
    for i = 1:M
        expression = expression + expand((x+y).^i);
    end
[notrequired, terms] = coeffs(expression, [x,y]);   

disp(terms);

phi=zeros(D,1);

w1=zeros(D,1);
w1_temp=zeros(D,1);

w2=zeros(D,1);
w2_temp=zeros(D,1);

w3=zeros(D,1);
w3_temp=zeros(D,1);

for i=1:D 
    w1(i)=2;w2(i)=2;w3(i)=2;
end

learning_rate=0.01;
delta=1000;


for i=1:13464
  X1(i,:)=transformed_vec(i,:);
end

for i=13465:26424
  X2(i-13464,:)=transformed_vec(i,:);
end

for i=26425:36936
  X3(i-26424,:)=transformed_vec(i,:);
end
%%%%%%%%%%%%%%%%%%%%%%Determining w1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while(delta>0)
    
phi_f=zeros(D,1);

for i=1:D 
    w1_temp(i)=w1(i);
end

% Training data of class 1
for i=1:Ntrain1
  arr=zeros(3,1);
  phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X1(i,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp((w1.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-1;
  
  phi=const3*phi;
  phi_f=plus(phi,phi_f);
end


% Training data of class 2
for i=1:Ntrain2
  arr=zeros(3,1);
   phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X2(i,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp(arr(1)-k);
  
  const21=exp(arr(1)-k);
  const22=exp(arr(2)-k);
  const23=exp(arr(3)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

% Training data of class 3
for i=1:Ntrain3
   arr=zeros(3,1);
   
   phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X3(i,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp(arr(1)-k);
  
  const21=exp(arr(1)-k);
  const22=exp(arr(2)-k);
  const23=exp(arr(3)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-0;

  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

phi_f=(learning_rate)*phi_f;
w1=minus(w1,phi_f);

%delta = sum(abs(w1-w1_temp));
delta=delta-1;
disp(delta);
end
disp('Completed 1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%Determining w2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta=1000;
while(delta>0)   
phi_f=zeros(D,1);
for i=1:D 
    w2_temp(i)=w2(i);
end
%%Training data of class 1
for i=1:Ntrain1
  arr=zeros(3,1);
  
   phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X1(i,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp(arr(2)-k);
  
  const21=exp(arr(1)-k);
  const22=exp(arr(2)-k);
  const23=exp(arr(3)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi,phi_f);
end


%%Training data of class 2
for i=1:Ntrain2
  arr=zeros(3,1);
  
  phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X2(i,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp(arr(2)-k);
  
  const21=exp(arr(1)-k);
  const22=exp(arr(2)-k);
  const23=exp(arr(3)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-1;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

%Training data of class 3
for i=1:Ntrain3
    arr=zeros(3,1);
  
     phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X3(i,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp(arr(2)-k);
  
  const21=exp(arr(1)-k);
  const22=exp(arr(2)-k);
  const23=exp(arr(3)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-0;

  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

phi_f=(learning_rate)*phi_f;
w2=minus(w2,phi_f);

%delta = sum(abs(w2-w2_temp));
delta=delta-1;
disp(delta);
end
disp('Completed 2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%Determining w3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta=1000;
while(delta>0)   
phi_f=zeros(D,1);
for i=1:D 
    w3_temp(i)=w3(i);
end
%Training data of class 1
for i=1:Ntrain1
    arr=zeros(3,1);
    
   phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X1(i,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp(arr(3)-k);
  
  const21=exp(arr(1)-k);
  const22=exp(arr(2)-k);
  const23=exp(arr(3)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi,phi_f);
end


%Training data of class 2
for i=1:Ntrain2
  arr=zeros(3,1);
  
   phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X2(i,k),pol(j,k));
    end
  end
  
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp(arr(3)-k);
  
  const21=exp(arr(1)-k);
  const22=exp(arr(2)-k);
  const23=exp(arr(3)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

%Training data of class 3
for i=1:Ntrain3
  arr=zeros(3,1);
  
  phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(X3(i,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  const1=exp(arr(3)-k);
  
  const21=exp(arr(1)-k);
  const22=exp(arr(2)-k);
  const23=exp(arr(3)-k);
  
  const2=const21+const22+const23;
  
  const3=(const1/const2)-1;

  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

phi_f=(learning_rate)*phi_f;
w3=minus(w3,phi_f);

%delta = sum(abs(w3-w3_temp));
delta=delta-1;
disp(delta);
end
disp('Completed 3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TESTING DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predicted1=zeros(Ntest1,1);
predicted2=zeros(Ntest2,1);
predicted3=zeros(Ntest3,1);

mt='..\data_assign3_group5\image_dataset\image_dataset\Features\mountain\test\';
Files=dir('..\data_assign3_group5\image_dataset\image_dataset\Features\mountain\test\');  
for t=3:length(Files)
  FileNames=Files(t).name;
  temp=strcat(mt,FileNames);
  temp111=importdata(temp);
  temp111=temp111*eigen(:,[1,2]);
  [ss,nq]=size(temp111);
  
  prob=zeros(3,1);
  
  for i=1:ss
     x_temp=temp111(i,:);
     arr=zeros(3,1);
     
    phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(x_temp(1,k),pol(j,k));
    end
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
  
  a1=exp(arr(1)-k);
  a2=exp(arr(2)-k);
  a3=exp(arr(3)-k);
  
  prob(1)=prob(1)+log(a1)-log(a1+a2+a3);
  prob(2)=prob(2)+log(a2)-log(a1+a2+a3);
  prob(3)=prob(3)+log(a3)-log(a1+a2+a3);
  end
  
  [mm,ind]=max(prob);
  predicted1(t-2,1)=ind;
end

%class2
cs='..\data_assign3_group5\image_dataset\image_dataset\Features\coast\test\';
Files=dir('..\data_assign3_group5\image_dataset\image_dataset\Features\coast\test\');  
for t=3:length(Files)
  FileNames=Files(t).name;
  temp=strcat(cs,FileNames);
  temp111=importdata(temp);
  temp111=temp111*eigen(:,[1,2]);
  [ss,nq]=size(temp111);
  
  prob=zeros(3,1);
  phi=ones(D,1);
  
  for i=1:ss
     x_temp=temp111(i,:);
     arr=zeros(3,1);
      phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(x_temp(1,k),pol(j,k));
    end
  end
    
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;
 
  a1=exp(arr(1)-k);
  a2=exp(arr(2)-k);
  a3=exp(arr(3)-k);
  
  prob(1)=prob(1)+log(a1)-log(a1+a2+a3);
  prob(2)=prob(2)+log(a2)-log(a1+a2+a3);
  prob(3)=prob(3)+log(a3)-log(a1+a2+a3);
  end
  [mm,ind]=max(prob);
  predicted2(t-2,1)=ind;
end

%class 3
strt='..\data_assign3_group5\image_dataset\image_dataset\Features\street\test\';
Files=dir('..\data_assign3_group5\image_dataset\image_dataset\Features\street\test\');  
for t=3:length(Files)
  FileNames=Files(t).name;
  temp=strcat(strt,FileNames);
  temp111=importdata(temp);
  temp111=temp111*eigen(:,[1,2]);
  [ss,nq]=size(temp111);
  
  prob=zeros(3,1);
  phi=ones(D,1);
  
  for i=1:ss
     x_temp=temp111(i,:);
     arr=zeros(3,1);
      phi=ones(D,1);
 
  for j=1:D
    for k=1:M
        phi(j)=phi(j)*power(x_temp(1,k),pol(j,k));      
    end
  end
    
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  
  [k,index]=max(arr) ;

  a1=exp(arr(1)-k);
  a2=exp(arr(2)-k);
  a3=exp(arr(3)-k);
    
  prob(1)=prob(1)+log(a1)-log(a1+a2+a3);
  prob(2)=prob(2)+log(a2)-log(a1+a2+a3);
  prob(3)=prob(3)+log(a3)-log(a1+a2+a3);
  end
  [mm,ind]=max(prob);
  predicted3(t-2,1)=ind;
end

%Computing confusion matrix.
confusion_matrix=zeros(3,3);
for i=1:Ntest1
  confusion_matrix(1,predicted1(i,1))=confusion_matrix(1,predicted1(i,1))+1;
end

for i=1:Ntest2
  confusion_matrix(2,predicted2(i,1))=confusion_matrix(2,predicted2(i,1))+1;
end

for i=1:Ntest3
  confusion_matrix(3,predicted3(i,1))=confusion_matrix(3,predicted3(i,1))+1;
end

disp(confusion_matrix);

accuracy=double((confusion_matrix(1,1)+confusion_matrix(2,2)+confusion_matrix(3,3)))/212;
disp(accuracy);