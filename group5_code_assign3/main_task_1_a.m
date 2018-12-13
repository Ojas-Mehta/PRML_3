clearvars;

x=dlmread('/home/cs18m041/ML/HandWritten_data/DATA/FeaturesHW/a.ldf');
y=dlmread('/home/cs18m041/ML/HandWritten_data/DATA/FeaturesHW/bA.ldf');
z=dlmread('/home/cs18m041/ML/HandWritten_data/DATA/FeaturesHW/chA.ldf');

alpha_1=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout1');
alpha_2=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout2');
alpha_3=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout3');

alpha_4=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout4');
alpha_5=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout5');
alpha_6=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout6');

alpha_7=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout7');
alpha_8=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout8');
alpha_9=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout9');
   
[size1,nq]=size(x);
[size2,nq]=size(y);
[size3,nq]=size(z);

train1_size=int16(0.8*size1);
train2_size=int16(0.8*size2);
train3_size=int16(0.8*size3);

test1_size=size1-train1_size;
test2_size=size2-train2_size;
test3_size=size3-train3_size;

c=15;

fp1=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/ap.txt','w+');
fp2=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/bp.txt','w+');
fp3=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/cp.txt','w+');
fp4=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/ap_test.txt','w+');
fp5=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/bp_test.txt','w+');
fp6=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/cp_test.txt','w+');

k=1;
%class 1%%
class1=[];
for i=1:train1_size
s1=x(i,1);
 for j=2:2:2*s1
     x_cor=x(i,j);
     y_cor=x(i,j+1);
     class1(k,1)=x_cor;
     class1(k,2)=y_cor;
     k=k+1;
 end
end

%class 2%%
for i=1:train2_size
s1=y(i,1);
 for j=2:2:2*s1
     x_cor=y(i,j);
     y_cor=y(i,j+1);
     class1(k,1)=x_cor;
     class1(k,2)=y_cor;
     k=k+1;
 end
end

for i=1:train3_size
s1=z(i,1);
 for j=2:2:2*s1
     x_cor=z(i,j);
     y_cor=z(i,j+1);
     class1(k,1)=x_cor;
     class1(k,2)=y_cor;
     k=k+1;
 end
end

[ind_k,centroid]=kmeans(class1,c);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TRAINING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

counter=1;

%%%class1%%%%%
for i=1:train1_size 
 s1=x(i,1);
 for j=1:s1
    index=ind_k(counter);
    fprintf(fp1,'%d ',(index-1));
    counter=counter+1;
 end
 fprintf(fp1,'\n');
end

%%%class2%%%%%
for i=1:train2_size 
 s1=y(i,1);
 for j=1:s1
    index=ind_k(counter);
    fprintf(fp2,'%d ',(index-1));
    counter=counter+1;
 end
 fprintf(fp2,'\n');
end

%%%class3%%%%%
for i=1:train3_size 
 s1=z(i,1);
 for j=1:s1
    index=ind_k(counter);
    fprintf(fp3,'%d ',(index-1));
    counter=counter+1;
 end
 fprintf(fp3,'\n');
end

counter=0;
%%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%

[len,nq]=size(centroid);

%class 1%%
for i=(train1_size+1):size1
 s1=x(i,1);
 for j=2:2:2*s1
     maxi=2000000;
     x_cor=x(i,j);
     y_cor=x(i,j+1);
     temp=zeros(1,2);
     temp(1,1)=x_cor;
     temp(1,2)=y_cor;
     for m=1:len
      dist=norm(centroid(m,:)-temp);
      if(dist<maxi)
          maxi=dist;
          index=m;
      end
     end
     fprintf(fp4,'%d ',(index-1)); 
 end
 fprintf(fp4,'\n');
end

%class 2%%
for i=(train2_size+1):size2
 s1=y(i,1);
 for j=2:2:2*s1
     maxi=2000000;
     x_cor=y(i,j);
     y_cor=y(i,j+1);
     temp=zeros(1,2);
     temp(1,1)=x_cor;
     temp(1,2)=y_cor;
     for m=1:len
      dist=norm(centroid(m,:)-temp);
      if(dist<maxi)
          maxi=dist;
          index=m;
      end
     end
     fprintf(fp5,'%d ',(index-1)); 
 end
 fprintf(fp5,'\n');
end

%class 3%%
for i=(train3_size+1):size3
 s1=z(i,1);
 for j=2:2:2*s1
     maxi=2000000;
     x_cor=z(i,j);
     y_cor=z(i,j+1);
     temp=zeros(1,2);
     temp(1,1)=x_cor;
     temp(1,2)=y_cor;
     for m=1:len
      dist=norm(centroid(m,:)-temp);
      if(dist<maxi)
          maxi=dist;
          index=m;
      end
     end
     fprintf(fp6,'%d ',(index-1)); 
 end
 fprintf(fp6,'\n');
end

%%%%%%%%%%%%%%%%%%%PREDICTION%%%%%%%%%%%%%%%%%
predicted1=zeros(train1_size,1);
predicted2=zeros(train2_size,1);
predicted3=zeros(train3_size,1);
count=0;

%class 1 prediction%
for i=1:train1_size
    prob=zeros(3,1);
    prob(1)=alpha_1(i);
    prob(2)=alpha_2(i);
    prob(3)=alpha_3(i);
    
    [mm,ind]=max(prob);
    predicted1(i,1)=ind;
    if(ind==1) count=count+1;end
end

%class 2 prediction%
for i=1:train2_size
    prob=zeros(3,1);
    prob(1)=alpha_4(i);
    prob(2)=alpha_5(i);
    prob(3)=alpha_6(i);
    
    [mm,ind]=max(prob);
    predicted2(i,1)=ind;
    if(ind==2) count=count+1;end
end

%class 3 prediction%
for i=1:train3_size
    prob=zeros(3,1);
    prob(1)=alpha_7(i);
    prob(2)=alpha_8(i);
    prob(3)=alpha_9(i);
    
    [mm,ind]=max(prob);
    predicted3(i,1)=ind;
    if(ind==3) count=count+1;end
end

accuracy=(double(count)*100/(train1_size+train2_size+train3_size));
disp(accuracy);

%%Computing confusion matrix

confusion_matrix=zeros(3,3);

for i=1:train1_size
 confusion_matrix(1,predicted1(i,1))=confusion_matrix(1,predicted1(i,1))+1;
end

for i=1:train2_size
 confusion_matrix(2,predicted2(i,1))=confusion_matrix(2,predicted2(i,1))+1;
end

for i=1:train3_size
 confusion_matrix(3,predicted3(i,1))=confusion_matrix(3,predicted3(i,1))+1;
end

disp(confusion_matrix);


%%%%%%%%%%%%%%%%%TRAINING ACCURACY AND CONFUSION%%%%%%%%%%%%%%%
% states:- 14 14
% 88.29%
% 80 1 0
% 3 65 10
% 4 10 66

%%%%%%%%%%%%%%%%%TESTING ACCURACY AND CONFUSION%%%%%%%%%%%%%%5
% 86%
% 20 0 0
% 0 14 15
% 0 3 17