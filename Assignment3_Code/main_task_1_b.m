clearvars;

c=12;

fp1=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/1p.txt','w+');
fp2=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/3p.txt','w+');
fp3=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/4p.txt','w+');
fp4=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/1p_test.txt','w+');
fp5=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/3p_test.txt','w+');
fp6=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/4p_test.txt','w+');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Training data%%%%%%%%%%%%%%%%%%%%%%%%%%%
class1=[];
string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/train/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/train/');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,' ',1,0);
   temp111(:,1)=[];
   class1=[class1;temp111];
end

string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/train/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/train/');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,' ',1,0);
   temp111(:,1)=[];
   class1=[class1;temp111];
end

string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/train/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/train/');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,' ',1,0);
   temp111(:,1)=[];
   class1=[class1;temp111];
end

[ind_k,centroid]=kmeans(class1,c);

counter=1;

string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/train/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/train/');

for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,'',1,0);
   [ss,nq]=size(temp111);
   for j=1:ss
    index=ind_k(counter,1);
    fprintf(fp1,'%d ',(index-1));
    counter=counter+1;
   end
   fprintf(fp1,'\n');
end

string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/train/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/train/');

for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,'',1,0);
   [ss,nq]=size(temp111);
   
   for j=1:ss
    index=ind_k(counter,1);
    fprintf(fp2,'%d ',(index-1));
    counter=counter+1;
   end
   fprintf(fp2,'\n');
end

string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/train/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/train/');

for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,'',1,0);
   [ss,nq]=size(temp111);
   
   for j=1:ss
    index=ind_k(counter,1);
    fprintf(fp3,'%d ',(index-1));
    counter=counter+1;
   end
   fprintf(fp3,'\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 [len,nq]=size(centroid);
 
string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/test/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/test/');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,' ',1,0);
   temp111(:,1)=[];
   [ss,nq]=size(temp111);
   
  
   index=0;
   
   for j=1:ss
       maxi=20000000;
     for m=1:len
      dist=norm(centroid(m,:)-temp111(j,:));
      if(dist<maxi)
          maxi=dist;
          index=m;
      end
     end
     
    fprintf(fp4,'%d ',(index-1));
    counter=counter+1;
   end
   
   fprintf(fp4,'\n');
end

string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/test/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/test/');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,' ',1,0);
   temp111(:,1)=[];
   [ss,nq]=size(temp111);
   
   maxi=20000000;
   index=0;
   
   for j=1:ss
        maxi=20000000;
     for m=1:len
      dist=norm(centroid(m,:)-temp111(j,:));
      if(dist<maxi)
          maxi=dist;
          index=m;
      end
     end
     
    fprintf(fp5,'%d ',(index-1));
    counter=counter+1;
   end
   
   fprintf(fp5,'\n');
end

string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/test/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/test/');
for k=3:length(Files)
   FileNames=Files(k).name;
   temp=strcat(string1,FileNames);
   temp111=dlmread(temp,' ',1,0);
   temp111(:,1)=[];
   [ss,nq]=size(temp111);

   maxi=20000000;
   index=0;
   
   for j=1:ss
        maxi=20000000;
     for m=1:len
      dist=norm(centroid(m,:)-temp111(j,:));
      if(dist<maxi)
          maxi=dist;
          index=m;
      end
     end
     
    fprintf(fp6,'%d ',(index-1));
    counter=counter+1;
   end
   
   fprintf(fp6,'\n');
end

% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 
%%Prediction%%
alpha_1=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout1');
alpha_2=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout2');
alpha_3=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout3');

alpha_4=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout4');
alpha_5=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout5');
alpha_6=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout6');

alpha_7=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout7');
alpha_8=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout8');
alpha_9=dlmread('/home/cs18m041/ML/hmm-1.04/hmm-1.04/alphaout9');

predicted1=zeros(12,1);
predicted2=zeros(12,1);
predicted3=zeros(12,1);
count=0;

%class 1 prediction%
for i=1:12
    prob=zeros(3,1);
    prob(1)=alpha_1(i);
    prob(2)=alpha_2(i);
    prob(3)=alpha_3(i);
    
    [mm,ind]=max(prob);
    predicted1(i,1)=ind;
    if(ind==1) count=count+1;end
end

%class 2 prediction%
for i=1:12
    prob=zeros(3,1);
    prob(1)=alpha_4(i);
    prob(2)=alpha_5(i);
    prob(3)=alpha_6(i);
    
    [mm,ind]=max(prob);
    predicted2(i,1)=ind;
    if(ind==2) count=count+1;end
end

%class 3 prediction%
for i=1:12
    prob=zeros(3,1);
    prob(1)=alpha_7(i);
    prob(2)=alpha_8(i);
    prob(3)=alpha_9(i);
    
    [mm,ind]=max(prob);
    predicted3(i,1)=ind;
    if(ind==3) count=count+1;end
end

accuracy=(double(count)/36)*100;
disp(accuracy);

%%%%%%%%%%%%%%%%%TRAINING ACCURACY AND CONFUSION%%%%%%%%%%%%%%%
% states:- 12 12
% 100%
% 45 0 0
% 0 45 0
% 0 0 45

%%%%%%%%%%%%%%%%%TESTING ACCURACY AND CONFUSION%%%%%%%%%%%%%%5
% 100%
% 12 0 0
% 0 12 0
% 0 0 12









