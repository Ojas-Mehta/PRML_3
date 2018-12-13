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

 M=3;
 d=2;
 syms x y;
 
 D = factorial(M+d) / (factorial(d)*factorial(M)); 
 [X,Y] = meshgrid(0:M);
 XY = [X(:),Y(:)];
 pol= zeros(D,d);
 len = length(XY);
 j=1;
 for i =1 : len
     if(XY(i,1)+XY(i,2) <= M)
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


ll=0.00005;
%disp(terms);


phi=zeros(D,1);

w1=zeros(D,1);
w1_temp=zeros(D,1);

w2=zeros(D,1);
w2_temp=zeros(D,1);

w3=zeros(D,1);
w3_temp=zeros(D,1);

w4=zeros(D,1);
w4_temp=zeros(D,1);

for i=1:D 
    w1(i)=2;w2(i)=2;w3(i)=2;w4(i)=2;
end

learning_rate=0.1;
delta=10;


%%%%%%%%%%%%%%%%%%%%%%Determining w1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while(delta>0.0001)
    
phi_f=zeros(D,1);

for i=1:D 
    w1_temp(i)=w1(i);
end

% Training data of class 1
for i=1:Ntrain1
  arr=zeros(4,1);
  for j = 1:D      
     phi(j)=power(x11_train(i),pol(j,1))*power(x12_train(i),pol(j,2));
  end
 
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w1.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-1;
  
  phi=const3*phi;
  phi_f=plus(phi,phi_f);
end


% Training data of class 2
for i=1:Ntrain2
  arr=zeros(4,1);
  for j = 1:D      
   phi(j)=power(x21_train(i),pol(j,1))*power(x22_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w1.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

% Training data of class 3
for i=1:Ntrain3
  arr=zeros(4,1);
  for j = 1:D      
     phi(j)=power(x31_train(i),pol(j,1))*power(x32_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
    
  [k,index]=max(arr);
  
  const1=exp((w1.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;

  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

% Training data of class 4
for i=1:Ntrain4
  arr=zeros(4,1);  
  for j = 1:D      
      phi(j)=power(x41_train(i),pol(j,1))*power(x42_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w1.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

phi_f=(ll)*phi_f;
w1=minus(w1,phi_f);

delta = sum(abs(w1-w1_temp));
disp(delta);
end
disp('Completed 1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%Determining w2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta=10;
while(delta>0.0001)   
phi_f=zeros(D,1);
for i=1:D 
    w2_temp(i)=w2(i);
end
%%Training data of class 1
for i=1:Ntrain1
  arr=zeros(4,1);
  for j = 1:D      
     phi(j)=power(x11_train(i),pol(j,1))*power(x12_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w2.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi,phi_f);
end


%%Training data of class 2
for i=1:Ntrain2
  arr=zeros(4,1);
  for j = 1:D      
   phi(j)=power(x21_train(i),pol(j,1))*power(x22_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w2.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-1;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

%Training data of class 3
for i=1:Ntrain3
    arr=zeros(4,1);
  for j = 1:D      
     phi(j)=power(x31_train(i),pol(j,1))*power(x32_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w2.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;

  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

%Training data of class 4
for i=1:Ntrain4
  arr=zeros(4,1);
  for j = 1:D      
      phi(j)=power(x41_train(i),pol(j,1))*power(x42_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w2.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

phi_f=(ll)*phi_f;
w2=minus(w2,phi_f);

delta = sum(abs(w2-w2_temp));
disp(delta);
end
disp('Completed 2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%Determining w3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta=10;
while(delta>0.0001)   
phi_f=zeros(D,1);
for i=1:D 
    w3_temp(i)=w3(i);
end
%Training data of class 1
for i=1:Ntrain1
    arr=zeros(4,1);
  for j = 1:D      
     phi(j)=power(x11_train(i),pol(j,1))*power(x12_train(i),pol(j,2));
  end
 
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w3.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi,phi_f);
end


%Training data of class 2
for i=1:Ntrain2
  arr=zeros(4,1);
  for j = 1:D      
   phi(j)=power(x21_train(i),pol(j,1))*power(x22_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w3.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

%Training data of class 3
for i=1:Ntrain3
  arr=zeros(4,1);
  for j = 1:D      
     phi(j)=power(x31_train(i),pol(j,1))*power(x32_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w3.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-1;

  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

%Training data of class 4
for i=1:Ntrain4
    arr=zeros(4,1);
  for j = 1:D      
      phi(j)=power(x41_train(i),pol(j,1))*power(x42_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w3.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

phi_f=(ll)*phi_f;
w3=minus(w3,phi_f);

delta = sum(abs(w3-w3_temp));
disp(delta);
end
disp('Completed 3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%Determining w4%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta=10;
while(delta>0.0001)   
phi_f=zeros(D,1);
for i=1:D 
    
    w4_temp(i)=w4(i);
end

%Training data of class 1
for i=1:Ntrain1
    arr=zeros(4,1);
  for j = 1:D      
     phi(j)=power(x11_train(i),pol(j,1))*power(x12_train(i),pol(j,2));
  end
 
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w4.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;

  phi=const3*phi;
  phi_f=plus(phi,phi_f);
end


%Training data of class 2
for i=1:Ntrain2
    arr=zeros(4,1);
  for j = 1:D      
   phi(j)=power(x21_train(i),pol(j,1))*power(x22_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w4.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;

  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

%Training data of class 3
for i=1:Ntrain3
    arr=zeros(4,1);
  for j = 1:D      
     phi(j)=power(x31_train(i),pol(j,1))*power(x32_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w4.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-0;
  
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

%Training data of class 4
for i=1:Ntrain4
    arr=zeros(4,1);
  for j = 1:D      
      phi(j)=power(x41_train(i),pol(j,1))*power(x42_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  const1=exp((w4.'*phi)-k);
  
  const21=exp((w1.'*phi)-k);
  const22=exp((w2.'*phi)-k);
  const23=exp((w3.'*phi)-k);
  const24=exp((w4.'*phi)-k);
  
  const2=const21+const22+const23+const24;
  
  const3=(const1/const2)-1;
 
  phi=const3*phi;
  phi_f=plus(phi_f,phi);
end

phi_f=(ll)*phi_f;
w4=minus(w4,phi_f);
delta = sum(abs(w4-w4_temp));
disp(delta);
end
disp('Completed 4');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TESTING DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predicted1=zeros(Ntest1,1);
predicted2=zeros(Ntest2,1);
predicted3=zeros(Ntest3,1);
predicted4=zeros(Ntest4,1);

for i=1:Ntest1
  prob=zeros(4,1);
  arr=zeros(4,1);
  
  for j = 1:D      
      phi(j)=power(x11_test(i),pol(j,1))*power(x12_test(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  a1=exp((w1.'*phi)-k);
  a2=exp((w2.'*phi)-k);
  a3=exp((w3.'*phi)-k);
  a4=exp((w4.'*phi)-k);
  
  prob(1)=a1/(a1+a2+a3+a4);
  prob(2)=a2/(a1+a2+a3+a4);
  prob(3)=a3/(a1+a2+a3+a4);
  prob(4)=a4/(a1+a2+a3+a4);
  
  [mm,ind]=max(prob);
  predicted1(i,1)=ind;
end

for i=1:Ntest2
  prob=zeros(4,1);
  arr=zeros(4,1);
  
  for j = 1:D      
      phi(j)=power(x21_test(i),pol(j,1))*power(x22_test(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  a1=exp((w1.'*phi)-k);
  a2=exp((w2.'*phi)-k);
  a3=exp((w3.'*phi)-k);
  a4=exp((w4.'*phi)-k);
  
  prob(1)=a1/(a1+a2+a3+a4);
  prob(2)=a2/(a1+a2+a3+a4);
  prob(3)=a3/(a1+a2+a3+a4);
  prob(4)=a4/(a1+a2+a3+a4);
  
  [mm,ind]=max(prob);
  predicted2(i,1)=ind;
end

for i=1:Ntest3
  prob=zeros(4,1);
  arr=zeros(4,1);
  
  for j = 1:D      
      phi(j)=power(x31_test(i),pol(j,1))*power(x32_test(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  a1=exp((w1.'*phi)-k);
  a2=exp((w2.'*phi)-k);
  a3=exp((w3.'*phi)-k);
  a4=exp((w4.'*phi)-k);
  
  prob(1)=a1/(a1+a2+a3+a4);
  prob(2)=a2/(a1+a2+a3+a4);
  prob(3)=a3/(a1+a2+a3+a4);
  prob(4)=a4/(a1+a2+a3+a4);
  
  [mm,ind]=max(prob);
  predicted3(i,1)=ind;
end

for i=1:Ntest4
  prob=zeros(4,1);
  arr=zeros(4,1);
  
  for j = 1:D      
      phi(j)=power(x41_test(i),pol(j,1))*power(x42_test(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  a1=exp((w1.'*phi)-k);
  a2=exp((w2.'*phi)-k);
  a3=exp((w3.'*phi)-k);
  a4=exp((w4.'*phi)-k);
  
  prob(1)=a1/(a1+a2+a3+a4);
  prob(2)=a2/(a1+a2+a3+a4);
  prob(3)=a3/(a1+a2+a3+a4);
  prob(4)=a4/(a1+a2+a3+a4);
  
  [mm,ind]=max(prob);
  predicted4(i,1)=ind;
end

x111=zeros(Ntrain1,1);
x112=zeros(Ntrain1,1);

for i=1:Ntrain1
    prob=zeros(4,1);
  arr=zeros(4,1);
  
  for j = 1:D      
      phi(j)=power(x11_train(i),pol(j,1))*power(x12_train(i),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  a1=exp((w1.'*phi)-k);
  a2=exp((w2.'*phi)-k);
  a3=exp((w3.'*phi)-k);
  a4=exp((w4.'*phi)-k);
  
  prob(1)=a1/(a1+a2+a3+a4);
  prob(2)=a2/(a1+a2+a3+a4);
  prob(3)=a3/(a1+a2+a3+a4);
  prob(4)=a4/(a1+a2+a3+a4);
  
  [mm,ind]=max(prob);
  if(ind==1)
      x111(i)=x11_train(i);
      x112(i)=x12_train(i);
  end
end

%Computing confusion matrix.
confusion_matrix=zeros(4,4);
for i=1:Ntest1
  confusion_matrix(1,predicted1(i,1))=confusion_matrix(1,predicted1(i,1))+1;
end

for i=1:Ntest2
  confusion_matrix(2,predicted2(i,1))=confusion_matrix(2,predicted2(i,1))+1;
end

for i=1:Ntest3
  confusion_matrix(3,predicted3(i,1))=confusion_matrix(3,predicted3(i,1))+1;
end

for i=1:Ntest4
  confusion_matrix(4,predicted4(i,1))=confusion_matrix(4,predicted4(i,1))+1;
end
disp(confusion_matrix);

accuracy=(confusion_matrix(1,1)+confusion_matrix(2,2)+confusion_matrix(3,3)+confusion_matrix(4,4))/400;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%PLOT%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xrange = [-15 20];
yrange = [-15 20];
inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
Ntest=length(xy);

predicted5=zeros(Ntest,4);
predicted=zeros(Ntest,1);

for i=1:Ntest
  x=xy(i,:);
  prob=zeros(4,1);
  arr=zeros(4,1);
  
  for j = 1:D      
      phi(j)=power(x(1,1),pol(j,1))*power(x(1,2),pol(j,2));
  end
  
  arr(1)=w1.'*phi;
  arr(2)=w2.'*phi;
  arr(3)=w3.'*phi;
  arr(4)=w4.'*phi;
  
  [k,index]=max(arr);
  
  a1=exp((w1.'*phi)-k);
  a2=exp((w2.'*phi)-k);
  a3=exp((w3.'*phi)-k);
  a4=exp((w4.'*phi)-k);
  
  prob(1)=a1/(a1+a2+a3+a4);
  prob(2)=a2/(a1+a2+a3+a4);
  prob(3)=a3/(a1+a2+a3+a4);
  prob(4)=a4/(a1+a2+a3+a4);
  
  [mm,ind]=max(prob);
  predicted5(i,1)=ind;
  predicted(i,1)=ind;
end

decisionmap = reshape(predicted, image_size);

figure;
imagesc(xrange,yrange,decisionmap);

hold on

set(gca,'ydir','normal');
cmap = [0.7 1 0.7;0.9 0.9 1;1 0.7 1;1 0.8 0.8];
colormap(cmap);

scatter(x111,x112,'g');
scatter(x21_train,x22_train,'b');
scatter(x31_train,x32_train,'m');
scatter(x41_train,x42_train,'r');
hold off

