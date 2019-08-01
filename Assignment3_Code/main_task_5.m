
clearvars;
[xtrain1] = textread('.\linearly_separable\class1_train.txt');
[xtrain2] = textread('.\linearly_separable\class2_train.txt');
[xtrain3] = textread('.\linearly_separable\class3_train.txt');
[xtrain4] = textread('.\linearly_separable\class4_train.txt');


[Ntrain1, NotRequired] = size(xtrain1);
[Ntrain2, NotRequired] = size(xtrain2);
[Ntrain3, NotRequired] = size(xtrain3);
[Ntrain4, NotRequired] = size(xtrain4);



[xval1] = textread('.\linearly_separable\class1_val.txt');
[xval2] = textread('.\linearly_separable\class2_val.txt');
[xval3] = textread('.\linearly_separable\class3_val.txt');
[xval4] = textread('.\linearly_separable\class4_val.txt');


[Nval1, NotRequired] = size(xval1);
[Nval2, NotRequired] = size(xval2);
[Nval3, NotRequired] = size(xval3);
[Nval4, NotRequired] = size(xval4);


[xtest1] = textread('.\linearly_separable\class1_test.txt');
[xtest2] = textread('.\linearly_separable\class2_test.txt');
[xtest3] = textread('.\linearly_separable\class3_test.txt');
[xtest4] = textread('.\linearly_separable\class4_test.txt');


[Ntest1, NotRequired] = size(xtest1);
[Ntest2, NotRequired] = size(xtest2);
[Ntest3, NotRequired] = size(xtest3);
[Ntest4, NotRequired] = size(xtest4);


label_1 = zeros(Ntrain1,1) +1;
label_2 = zeros(Ntrain2,1)+2;
label_3 = zeros(Ntrain3,1)+3;
label_4 = zeros(Ntrain4,1)+4;

val_label_1 = zeros(Nval1,1) +1;
val_label_2 = zeros(Nval2,1)+2;
val_label_3 = zeros(Nval3,1)+3;
val_label_4 = zeros(Nval4,1)+4;

test_label_1 = zeros(Ntest1,1) +1;
test_label_2 = zeros(Ntest2,1)+2;
test_label_3 = zeros(Ntest3,1)+3;
test_label_4 = zeros(Ntest4,1)+4;

Ntrain = Ntrain1+Ntrain2+Ntrain3+Ntrain4;
Nval = Nval1+Nval2+Nval3+Nval4;
Ntest = Ntest1+Ntest2+Ntest3+Ntest4;


xtrain = [xtrain1; xtrain2; xtrain3; xtrain4];
xval = [xval1; xval2; xval3; xval4];
xtest = [xtest1; xtest2; xtest3; xtest4];

xlabel = [label_1; label_2; label_3; label_4];

val_label_check = [val_label_1; val_label_2; val_label_3; val_label_4];

test_label_check = [test_label_1; test_label_2; test_label_3; test_label_4];

weightC = 0.05;
xtrain = xtest;
Ntrain = Ntest;
Ntrain1 = Ntest1;
Ntrain2 = Ntest2;
Ntrain3 = Ntest3;
xlabel = test_label_check;
% xtrain = xval;
% Ntrain = Nval;
% Ntrain1 = Nval1;
% Ntrain2 = Nval2;
% Ntrain3 = Nval3;
% xlabel = val_label_check;



model = svmtrain(xlabel, xtrain, '-s 0 -t 0 -c 0.05');

[predict_train, train_accuracy, dec_values] = svmpredict(xlabel, xtrain, model);
% [predict_val, val_accuracy, dec_values] = svmpredict(val_label_check, xval, model);
% [predict_test, test_accuracy, dec_values] = svmpredict(test_label_check, xtest, model);

%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%
groundTruth = xlabel;
d = xtrain;

figure

% plot training data
hold on;
pos = find(groundTruth==1);
scatter(d(pos,1), d(pos,2), 'r')
pos = find(groundTruth==2);
scatter(d(pos,1), d(pos,2), 'b')
pos = find(groundTruth==3);
scatter(d(pos,1), d(pos,2), 'g')
pos = find(groundTruth==4);
scatter(d(pos,1), d(pos,2), 'm')

% now plot support vectors
hold on;
sv = full(model.SVs);

nsv = size(sv, 1);
svbounded = [];
svunbounded = [];

for i = 1:nsv
    if abs(model.sv_coef(i,1)) == weightC || abs(model.sv_coef(i,2)) == weightC || abs(model.sv_coef(i,3)) == weightC
        svbounded = [svbounded; sv(i,:)];
    else
        svunbounded = [svunbounded; sv(i,:)];
    end
end
if(size(svunbounded,1)>0)
plot(svunbounded(:,1),svunbounded(:,2),'o','MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 4);
end
if(size(svbounded,1)>0)
plot(svbounded(:,1),svbounded(:,2),'o','MarkerFaceColor', 'y', 'MarkerEdgeColor', 'y', 'MarkerSize', 4);
end
%plot(sv(:,1),sv(:,2),'kx','MarkerSize',12);

% now plot decision area
[xi,yi] = meshgrid([min(([d(:,1); d(:,2)])):0.01:max([d(:,1); d(:,2)])],[min(([d(:,1); d(:,2)])):0.01:max([d(:,1); d(:,2)])]);
dd = [xi(:),yi(:)];
tic;[predicted_label, accuracy, decision_values] = svmpredict(zeros(size(dd,1),1), dd, model);toc
pos = find(predicted_label==1);
hold on;
redcolor = [1 0.8 0.8];
h1 = plot(dd(pos,1),dd(pos,2),'s','color',redcolor,'MarkerSize',10,'MarkerEdgeColor',redcolor,'MarkerFaceColor',redcolor);
pos = find(predicted_label==2);
hold on;
bluecolor = [0.8 0.8 1];
h2 = plot(dd(pos,1),dd(pos,2),'s','color',bluecolor,'MarkerSize',10,'MarkerEdgeColor',bluecolor,'MarkerFaceColor',bluecolor);


pos = find(predicted_label==3);
hold on;
cyancolor = [0.6 1 0.6];
h3 = plot(dd(pos,1),dd(pos,2),'s','color',cyancolor,'MarkerSize',10,'MarkerEdgeColor',cyancolor,'MarkerFaceColor',cyancolor);
pos = find(predicted_label==4);
hold on;
magentacolor = [1 0.7 0.9];
h4 = plot(dd(pos,1),dd(pos,2),'s','color',magentacolor,'MarkerSize',10,'MarkerEdgeColor',magentacolor,'MarkerFaceColor',magentacolor);

uistack(h1, 'bottom');
uistack(h2, 'bottom');
uistack(h3, 'bottom');
uistack(h4, 'bottom');






conftotal = size(xlabel,1);
count = zeros(4, 4);
for i = 1:conftotal      
    count(xlabel(i), predict_train(i)) = count(xlabel(i), predict_train(i)) + 1;
end    
disp(count);
%model = svmtrain(training_label_vector, training_instance_matrix [, 'libsvm_options']);
%[predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model [, 'libsvm_options']);
% training_label_vector = [1 2];
% training_instance_matrix = [1 2; 3 4];
% testing_instance_matrix = [1 2; 3 4];
% testing_label_vector = [1 2];
% model = svmtrain(training_label_vector, training_instance_matrix);
% [predicted_label, accuracy] = svmpredict(testing_label_vector, testing_instance_matrix, model);

%[heart_scale_label, heart_scale_inst] = libsvmread('heart_scale');
%model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07');
%[predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model); % test the training data
% labels = double(rand(10,1)>0.5);
% data = rand(10,5);
% model = svmtrain(labels, data, '-s 0 -t 2 -c 1 -g 0.1');
% [predict_label, accuracy, dec_values] = svmpredict(labels, data, model);
%Smaller the C, Larger will be the margin