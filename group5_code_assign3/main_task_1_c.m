% 
states=12;
symbols=12;
%Generating two digit sequences%%

id=zeros(3,1);
id(1,1)=1;
id(2,1)=3;
id(3,1)=4;

for m=1:3
    for n=1:3
        counter=1;
        string1='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
        string1=strcat(string1,num2str(id(m,1)),'p','.txt.hmm');
        first=importdata(string1);
      
        string2='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
        string2=strcat(string2,num2str(id(n,1)),'p','.txt.hmm');
        second=importdata(string2);
        
        string3='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
        result=strcat(string3,num2str(id(m,1)),'p',num2str(id(n,1)),'p','.txt.hmm');
        fp1=fopen(result,'w+');
        %First HMM
        fprintf(fp1,'states: 25');fprintf(fp1,'\n');
        fprintf(fp1,'symbols: 12');fprintf(fp1,'\n');
        arr=first(1,1).data;
        for j=1:(states-1)
          for i=1:symbols+1
            fprintf(fp1,'%f ',arr(counter,i));
          end
          counter=counter+1;
          fprintf(fp1,'\n');
          for i=1:(symbols+1)
            fprintf(fp1,'%f ',arr(counter,i));
          end
          counter=counter+1;
          fprintf(fp1,'\n');
          fprintf(fp1,'\n');
        end

          for i=1:symbols+1
            fprintf(fp1,'%f ',arr(counter,i));
          end
          counter=counter+1;
          fprintf(fp1,'\n');
          fprintf(fp1,'0.150000 ');
          for i=2:(symbols+1)
            fprintf(fp1,'%f ',arr(counter,i));
          end
          counter=counter+1;
          fprintf(fp1,'\n');
          fprintf(fp1,'\n');
          
        %Adding extra state
        for i=1:(symbols+1)
         fprintf(fp1,'0.000 ');
        end
        fprintf(fp1,'\n');

        fprintf(fp1,'1.0000 ');
        for i=2:(symbols+1)
         fprintf(fp1,'0.5000 ');
        end
        fprintf(fp1,'\n');
        fprintf(fp1,'\n');

        %Second HMM%%%%%%%
        counter=1;
        arr=second(1,1).data;
        for j=1:states
          for i=1:symbols+1
            fprintf(fp1,'%f ',arr(counter,i));
          end
          counter=counter+1;
          fprintf(fp1,'\n');
          for i=1:(symbols+1)
            fprintf(fp1,'%f ',arr(counter,i));
          end
          counter=counter+1;
          fprintf(fp1,'\n');
          fprintf(fp1,'\n');
        end
    end 
end

%%%%%%%%%%%%%%%%%%Generating 3 length sequences%%%%%%%%%%%%%%%%%%%
for m=1:3
    for n=1:3
        for o=1:3
            counter=1;
            string1='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
            string1=strcat(string1,num2str(id(m,1)),'p','.txt.hmm');
            first=importdata(string1);

            string2='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
            string2=strcat(string2,num2str(id(n,1)),'p','.txt.hmm');
            second=importdata(string2);

            
            string3='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
            string3=strcat(string3,num2str(id(o,1)),'p','.txt.hmm');
            third=importdata(string3);
            
            string4='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
            result=strcat(string4,num2str(id(m,1)),'p',num2str(id(n,1)),'p',num2str(id(o,1)),'p','.txt.hmm');
            
            fp1=fopen(result,'w+');
            fprintf(fp1,'states: 39');fprintf(fp1,'\n');
            fprintf(fp1,'symbols: 12');fprintf(fp1,'\n');
            %First HMM
            arr=first(1,1).data;
            for j=1:(states-1)
              for i=1:symbols+1
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              for i=1:(symbols+1)
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              fprintf(fp1,'\n');
            end

              for i=1:symbols+1
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              fprintf(fp1,'0.150000 ');
              for i=2:(symbols+1)
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              fprintf(fp1,'\n');

            %Adding extra state
            for i=1:(symbols+1)
             fprintf(fp1,'0.00 ');
            end
            fprintf(fp1,'\n');

            fprintf(fp1,'1.00 ');
            for i=2:(symbols+1)
             fprintf(fp1,'0.500 ');
            end
            fprintf(fp1,'\n');
            fprintf(fp1,'\n');

            %Second HMM%%%%%%%
            counter=1;
            arr=second(1,1).data;
            for j=1:(states-1)
              for i=1:symbols+1
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              for i=1:(symbols+1)
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              fprintf(fp1,'\n');
            end

              for i=1:symbols+1
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              fprintf(fp1,'0.009000');
              for i=2:(symbols+1)
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              fprintf(fp1,'\n');
            
            %Adding extra state
            for i=1:(symbols+1)
             fprintf(fp1,'0.000 ');
            end
            fprintf(fp1,'\n');

            fprintf(fp1,'1.000 ');
            for i=2:(symbols+1)
             fprintf(fp1,'0.500 ');
            end
            fprintf(fp1,'\n');
            fprintf(fp1,'\n');
            
            %%Third HMM
            counter=1;
            arr=third(1,1).data;
            for j=1:states
              for i=1:symbols+1
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              for i=1:(symbols+1)
                fprintf(fp1,'%f ',arr(counter,i));
              end
              counter=counter+1;
              fprintf(fp1,'\n');
              fprintf(fp1,'\n');
            end
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%Training data%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c=12;
% 
% class1=[];
% size_count=0;
% 
% string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/train/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/train/');
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,' ',1,0);
%    temp111(:,1)=[];
%    size_count=size_count+size(temp111,1);
%    class1=[class1;temp111];
% end
% 
% string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/train/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/train/');
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,' ',1,0);
%    temp111(:,1)=[];
%    size_count=size_count+size(temp111,1);
%    class1=[class1;temp111];
% end
% 
% string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/train/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/train/');
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,' ',1,0);
%    temp111(:,1)=[];
%    size_count=size_count+size(temp111,1);
%    class1=[class1;temp111];
% end
% 
% 
% string1='/home/cs18m041/ML/speech_dataset/speech_data/connected/test1/5/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/connected/test1/5/');
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,' ',1,0);
%    temp111(:,1)=[];
%    class1=[class1;temp111];
% end
% 
% 
% fp10=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/1p.txt','w+');
% fp20=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/3p.txt','w+');
% fp30=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/4p.txt','w+');
% 
% [ind_k,centroid]=kmeans(class1,c);
% 
% %%Assigning cluster numbers
% counter=1;
% 
% string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/train/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/1/train/');
% 
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,'',1,0);
%    [ss,nq]=size(temp111);
%    for j=1:ss
%     index=ind_k(counter,1);
%     fprintf(fp10,'%d ',(index-1));
%     counter=counter+1;
%    end
%    fprintf(fp10,'\n');
% end
% 
% string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/train/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/3/train/');
% 
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,'',1,0);
%    [ss,nq]=size(temp111);
%    
%    for j=1:ss
%     index=ind_k(counter,1);
%     fprintf(fp20,'%d ',(index-1));
%     counter=counter+1;
%    end
%    fprintf(fp20,'\n');
% end
% 
% string1='/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/train/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/isolated/5/4/train/');
% 
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,'',1,0);
%    [ss,nq]=size(temp111);
%    
%    for j=1:ss
%     index=ind_k(counter,1);
%     fprintf(fp30,'%d ',(index-1));
%     counter=counter+1;
%    end
%    fprintf(fp30,'\n');
% end
% 
% string1='/home/cs18m041/ML/speech_dataset/speech_data/connected/test1/5/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/connected/test1/5');
% 
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,'',1,0);
%    [ss,nq]=size(temp111);
%    
%    string3='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
%    string4=strcat(string3,FileNames,'p.txt');
%    fp40=fopen(string4,'w+');
%    
%    for j=1:ss
%     index=ind_k(counter,1);
%     fprintf(fp40,'%d ',(index-1));
%     counter=counter+1;
%    end
%    fprintf(fp40,'\n');
% end

%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%%%
% [len,nq]=size(centroid);
%  
% string1='/home/cs18m041/ML/speech_dataset/speech_data/connected/test1/5/';
% Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/connected/test1/5/');
% for k=3:length(Files)
%    FileNames=Files(k).name;
%    temp=strcat(string1,FileNames);
%    temp111=dlmread(temp,' ',1,0);
%    temp111(:,1)=[];
%    [ss,nq]=size(temp111);
%    
%    string3='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
%    string4=strcat(string3,FileNames,'p.txt');
%    fp40=fopen(string4,'w+');
%    
%    index=0;   
%    for j=1:ss
%        maxi=20000000000;
%      for m=1:len
%       dist=norm(centroid(m,:)-temp111(j,:));
%       if(dist<maxi)
%           maxi=dist;
%           index=m;
%       end
%      end
%      
%     fprintf(fp40,'%d ',(index-1));
%     counter=counter+1;
%    end
%    fprintf(fp40,'\n');
% end