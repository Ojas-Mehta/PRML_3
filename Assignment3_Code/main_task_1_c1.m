fp10=fopen('/home/cs18m041/ML/hmm-1.04/hmm-1.04/testing_connected.sh','w+');

test_string='./test_hmm';
alpha_string='mv alphaout alphaout';
arr=zeros(3,1);
arr(1,1)=1;
arr(2,1)=3;
arr(3,1)=4;

counter=1;

map_counter=1;

mapping=zeros(39,1);

% hh=1;
% gg=2;
% mm=strcat(num2str(hh),num2str(gg));
% nn=str2double(mm);
% disp(nn);

string1='/home/cs18m041/ML/speech_dataset/speech_data/connected/test2/5/';
Files=dir('/home/cs18m041/ML/speech_dataset/speech_data/connected/test2/5/');
for k=15:15
   FileNames=Files(k).name;
   string4=strcat(FileNames,'p.txt');
   string4=[test_string ' ' string4];
   
   %%One-length sequences
   string5=[string4 ' ' '1p.txt.hmm'];
   fprintf(fp10,string5);fprintf(fp10,'\n');
   string100=strcat(alpha_string,num2str(counter));
   fprintf(fp10,string100);fprintf(fp10,'\n');
   counter=counter+1;
   mapping(map_counter,1)=1; map_counter=map_counter+1;
   
   string6=[string4 ' ' '3p.txt.hmm'];
   fprintf(fp10,string6);fprintf(fp10,'\n');
   string100=strcat(alpha_string,num2str(counter));
   fprintf(fp10,string100);fprintf(fp10,'\n');
   counter=counter+1;
   mapping(map_counter,1)=3; map_counter=map_counter+1;
   
   string7=[string4 ' ' '4p.txt.hmm'];
   fprintf(fp10,string7);fprintf(fp10,'\n');
   string100=strcat(alpha_string,num2str(counter));
   fprintf(fp10,string100);fprintf(fp10,'\n');
   counter=counter+1;
   mapping(map_counter,1)=4; map_counter=map_counter+1;
   
   fprintf(fp10,'\n');
   
   %%Two-length sequences
   for m=1:3
       for n=1:3
           string5=strcat(num2str(arr(m,1)),'p',num2str(arr(n,1)),'p.txt.hmm');
           string6=[string4 ' ' string5];
           fprintf(fp10,string6);fprintf(fp10,'\n');
           string100=strcat(alpha_string,num2str(counter));
           fprintf(fp10,string100);fprintf(fp10,'\n');
           counter=counter+1;
           
           string23=strcat(num2str(arr(m,1)),num2str(arr(n,1)));
           mapping(map_counter,1)=str2double(string23); map_counter=map_counter+1;
       end
   end
   
   fprintf(fp10,'\n');
   %%Three-length sequences
   for m=1:3
       for n=1:3
           for o=1:3
               string5=strcat(num2str(arr(m,1)),'p',num2str(arr(n,1)),'p',num2str(arr(o,1)),'p.txt.hmm');
               string6=[string4 ' ' string5];
               fprintf(fp10,string6);fprintf(fp10,'\n');
               string100=strcat(alpha_string,num2str(counter));
               fprintf(fp10,string100);fprintf(fp10,'\n');
               counter=counter+1;
               
               string23=strcat(num2str(arr(m,1)),num2str(arr(n,1)),num2str(arr(o,1)));
               mapping(map_counter,1)=str2double(string23); map_counter=map_counter+1;
           end
       end
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PREDICTION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

max=-1000000000000;

ind=0;
for i=1:39
 string1='/home/cs18m041/ML/hmm-1.04/hmm-1.04/';
 string2=strcat(string1,'alphaout',num2str(i));
 disp(string2);
 data=importdata(string2);
 if(data(1,1)>max)
   max=data(1,1);
   ind=i;
 end
end

disp(mapping(ind,1));

%%%%%%%%%%%%%%%PREDICTION FOR FILES%%%%%%%%%%%%
% 1) 133 134
% 2) 143 144
% 3) 143 14
% 4) 11  13
% 5) 313 334
% 6) 333 333
% 7) 344 344
% 8) 341 341
% 9) 134 431
% 10) 14 44
% 11) 434 143
% 12) 433 434
% 13) 443 44