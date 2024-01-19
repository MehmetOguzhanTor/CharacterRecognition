
clear all;
close all;
clc;

load digits.mat
i = 1; %select the digit to display
I = digits( i, : );
imagesc( reshape( I, 20, 20 ) );
colormap( gray );
axis image;

%% Question 1
%Shuffiling the dataset
ind = [1:5000];
ind = randperm(length(ind));

%spliting dataset and labels into train and test sets equally
train_ind = ind([1:2500]);
test_ind = ind([2501:5000]);
train_digits = digits(train_ind,:);
test_digits = digits(test_ind,:);
train_labels = labels(train_ind);
test_labels = labels(test_ind);

%calculating the mean, std and variance of the trainset of digits dataset
mean_digits = mean(train_digits);
std_digits = std(train_digits);
var_digits = std_digits'*std_digits;

%finding the cent and covarience arrays 
cent_array = train_digits - mean_digits;
cov_array = cent_array'*cent_array;

%finding vectors and eigen values by using eig function
[vectors, eigen_values] = eig(cov_array);

%diagonalizing eigen values
eigen_values = diag(eigen_values);

%sorting eigen values
[sorted_eigens, index] = sort(eigen_values);

%converting eigen values into decsecding order
flipped_eigens = flip(eigen_values);
flipped_index = flip(index);

%plotting the eigenvalues in descending order as required in the assignment
figure;
stem(flipped_eigens);
title('Eigen Values in descending order')

%Eigen vectors in the same order with eigenvalues
vector = [];
vector = vectors(:,flipped_index);

%Displaing eigen vectors using the function disp_eig
disp_eig(vector)

%we would choose 50 of them because after 50 the values descent slowly
pca_result = train_digits*vector(:,1:50);

%we get faded images
im = reshape(vector(:,50), [20,20]);
figure;
imshow(im);

%Reconstraction the vector
reconstract = pca_result*vector(:,1:50)';
im2 = reshape(reconstract(5,:), [20,20]);

%Displaying the mean digits
im3 = reshape(mean_digits, [20,20]);

%% Question 1.3 and 1.4
%Using functions of gauss_train and gauss_test for train and test sets
%inside a for look for different subsets
iter = 1
for i = 20:5:120
    [mean_cell, cov_cell] = gauss_train(train_digits,train_labels,vector,i);
    accuracy_test = gauss_test(test_digits, test_labels, vector, mean_cell,cov_cell,i);
    acc_test(iter) = accuracy_test;
    accuracy_train = gauss_test(train_digits, train_labels, vector, mean_cell,cov_cell,i);
    acc_train(iter) = accuracy_train;
    iter = iter +1;
end

%Plotting the accuracies for the train and the test set
figure;
plot([20:5:120],acc_train);
xlabel('Number of Dimensions')
ylabel('Accuracy')
title('Gaussian Classification Performed with PCA')
hold on;
plot([20:5:120],acc_test);
hold off;


%% Question 2.1

% Using LDA function to obtain eigen vectors
[Y, W, lambda] = LDA(train_digits, train_labels);
disp_eig(W)

%% Question 2.2 and 2.3
%Using functions of gauss_train and gauss_test for train and test sets
%inside a for look for different subsets
for i = 1:20
    [mean_cell, cov_cell] = gauss_train(train_digits,train_labels,W,i);
    accuracy_test_LDA = gauss_test(test_digits, test_labels, W, mean_cell,cov_cell,i);
    acc_test_LDA(i) = accuracy_test_LDA;
    accuracy_train_LDA = gauss_test(train_digits, train_labels, W, mean_cell,cov_cell,i);
    acc_train_LDA(i) = accuracy_train_LDA;
end

%Plotting the accuracies for the train and the test set
figure;
plot(acc_train_LDA);
xlabel('Number of Dimensions')
ylabel('Accuracy')
title('Gaussian Classification Performed with LDA')
hold on;
plot(acc_test_LDA);
hold off;


%% Q3
%Here we can change the options in the Sammon's mapping function
%MaxIter is decreased to make the code run faster
opts.Display        = 'iter';
opts.Input          = 'raw';
opts.MaxHalves      = 20;
opts.MaxIter        = 10;
opts.TolFun         = 1e-9;
opts.Initialisation = 'random';

%Using Sammon's mapping
[y, E] = sammon(digits, 2, opts);

%Using tsne
Y = tsne(digits);

%Plotting the results of Sammon's and tsne mapping
figure;
scatter(Y(:,1),Y(:,2));
title('Sammons Mapping')
figure;
scatter(y(:,1),y(:,2));
title('tsne Mapping')


%% Functions

% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML114
% Project Title: Implementation of Linear Discriminant Analysis in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com

function [Y, W, lambda] = LDA(X, L)
    Classes=unique(L)';
    k=numel(Classes);
    n=zeros(k,1);
    C=cell(k,1);
    M=mean(X);
    S=cell(k,1);
    Sw=0;
    Sb=0;
    for j=1:k
        Xj=X(L==Classes(j),:);
        n(j)=size(Xj,1);
        C{j}=mean(Xj);
        S{j}=0;
        for i=1:n(j)
            S{j}=S{j}+(Xj(i,:)-C{j})'*(Xj(i,:)-C{j});
        end
        Sw=Sw+S{j};
        Sb=Sb+n(j)*(C{j}-M)'*(C{j}-M);
    end
    [W, LAMBDA]=eig(Sb,Sw);
    lambda=diag(LAMBDA);
    [lambda, SortOrder]=sort(lambda,'descend');
    W=W(:,SortOrder);
    Y=X*W;
end

function [mean_cell, cov_cell] = gauss_train(train_digits,train_labels,vector,dim)
%Training the given data for every class
projected_data = train_digits*vector(:,[1:dim]);
zero_index=[];one_index=[];two_index=[];three_index=[];four_index=[];
five_index=[];six_index=[];seven_index=[];eight_index=[];nine_index=[];

for i=1:size(train_labels,1)
    if train_labels(i) == 0
        zero_index = [zero_index i];
    elseif train_labels(i) == 1
        one_index = [one_index i];
    elseif train_labels(i) == 2
        two_index = [two_index i];
    elseif train_labels(i) == 3
        three_index = [three_index i];
    elseif train_labels(i) == 4
        four_index = [four_index i];
    elseif train_labels(i) == 5
        five_index = [five_index i];
    elseif train_labels(i) == 6
        six_index = [six_index i];
    elseif train_labels(i) == 7
        seven_index = [seven_index i];
    elseif train_labels(i) == 8
        eight_index = [eight_index i];
    elseif train_labels(i) == 9
        nine_index = [nine_index i];
    end
end
class_index = {zero_index,one_index,two_index,three_index,four_index,...
    five_index,six_index,seven_index,eight_index,nine_index};

for c=1:10
    mean_cell{c} = mean(projected_data(class_index{c},:));
    cov_cell{c} =cov(projected_data(class_index{c},:));
end
end

function accuracy = gauss_test(test_digits, test_labels, vector, mean_cell,cov_cell,dim)
%gauss test function finds the prediction and the accuracy percentage
projected_test = test_digits*vector(:,[1:dim]);

prediction = [];
for t = 1:2500
    for c = 1:10
        e = (-0.5)*(projected_test(t,:) - mean_cell{c})*((cov_cell{c})^(-1))*(projected_test(t,:) - mean_cell{c})';
        a = ((2*pi)^(size(projected_test,2)));
        b = sqrt(norm(cov_cell{c}));
        probs(c) = (1/(a*b))*(exp(e));
    end
    [M,predict] = max(probs);
    prediction = [prediction predict];
end
prediction = prediction-1;
accuracy = sum(prediction'==test_labels)/2500;
accuracy = 100*accuracy;
end

function disp_eig(vector)
%dip_eig function displays the vectors that is given as input
k = 1;
for i= 1:size(vector,1)
    img_cell{k,mod(i,20)+1} = reshape(vector(:,i),[20,20]);
    if mod(i,20) == 0
        k = k+1;
    end
end
vector_img = cell2mat(img_cell);
figure;
imagesc(vector_img);
colormap('gray')
% colormap('gray');
end

function [y, E] = sammon(x, n, opts)
%SAMMON Performs Sammon's MDS mapping on dataset X
%
%    Y = SAMMON(X) applies Sammon's nonlinear mapping procedure on
%    multivariate data X, where each row represents a pattern and each column
%    represents a feature.  On completion, Y contains the corresponding
%    co-ordinates of each point on the map.  By default, a two-dimensional
%    map is created.  Note if X contains any duplicated rows, SAMMON will
%    fail (ungracefully). 
%
%    [Y,E] = SAMMON(X) also returns the value of the cost function in E (i.e.
%    the stress of the mapping).
%
%    An N-dimensional output map is generated by Y = SAMMON(X,N) .
%
%    A set of optimisation options can also be specified using a third
%    argument, Y = SAMMON(X,N,OPTS) , where OPTS is a structure with fields:
%
%       MaxIter        - maximum number of iterations
%       TolFun         - relative tolerance on objective function
%       MaxHalves      - maximum number of step halvings
%       Input          - {'raw','distance'} if set to 'distance', X is 
%                        interpreted as a matrix of pairwise distances.
%       Display        - {'off', 'on', 'iter'}
%       Initialisation - {'pca', 'random'}
%
%    The default options structure can be retrieved by calling SAMMON with
%    no parameters.
%
%    References :
%
%       [1] Sammon, John W. Jr., "A Nonlinear Mapping for Data Structure
%           Analysis", IEEE Transactions on Computers, vol. C-18, no. 5,
%           pp 401-409, May 1969.
%
%    See also : SAMMON_TEST

%
% File        : sammon.m
%
% Date        : Monday 12th November 2007.
%
% Author      : Gavin C. Cawley and Nicola L. C. Talbot
%
% Description : Simple vectorised MATLAB implementation of Sammon's non-linear
%               mapping algorithm [1].
%
% References  : [1] Sammon, John W. Jr., "A Nonlinear Mapping for Data
%                   Structure Analysis", IEEE Transactions on Computers,
%                   vol. C-18, no. 5, pp 401-409, May 1969.
%
% History     : 10/08/2004 - v1.00
%               11/08/2004 - v1.10 Hessian made positive semidefinite
%               13/08/2004 - v1.11 minor optimisation
%               12/11/2007 - v1.20 initialisation using the first n principal
%                                  components.
%
% Thanks      : Dr Nick Hamilton (nick@maths.uq.edu.au) for supplying the
%               code for implementing initialisation using the first n
%               principal components (introduced in v1.20).
%
% To do       : The current version does not take advantage of the symmetry
%               of the distance matrix in order to allow for easy
%               vectorisation.  This may not be a good choice for very large
%               datasets, so perhaps one day I'll get around to doing a MEX
%               version using the BLAS library etc. for very large datasets.
%
% Copyright   : (c) Dr Gavin C. Cawley, November 2007.
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology


    % use the default options structure
    if nargin < 3
        opts.Display        = 'iter';
        opts.Input          = 'raw';
        opts.MaxHalves      = 20;
        opts.MaxIter        = 500;
        opts.TolFun         = 1e-9;
        opts.Initialisation = 'random';

    end

    % the user has requested the default options structure
    if nargin == 0
        y = opts;
        return;
    end

    % Create a two-dimensional map unless dimension is specified
    if nargin < 2
        n = 2;
    end

    % Set level of verbosity
    if strcmp(opts.Display, 'iter')
        display = 2;
    elseif strcmp(opts.Display, 'on')
        display = 1;
    else
        display = 0;
    end

    % Create distance matrix unless given by parameters
    if strcmp(opts.Input, 'distance')
        D = x;
    else
        D = euclid(x, x);
    end

    % Remaining initialisation
    N     = size(x, 1);
    scale = 0.5 / sum(D(:));
    D     = D + eye(N);
    Dinv  = 1 ./ D;
    if strcmp(opts.Initialisation, 'pca')
        [UU,DD] = svd(x);
        y       = UU(:,1:n)*DD(1:n,1:n);
    else
        y = randn(N, n);
    end
    one   = ones(N,n);
    d     = euclid(y,y) + eye(N);
    dinv  = 1./d;
    delta = D - d;
    E     = sum(sum((delta.^2).*Dinv));

    % Get on with it
    for i=1:opts.MaxIter

        % Compute gradient, Hessian and search direction (note it is actually
        % 1/4 of the gradient and Hessian, but the step size is just the ratio
        % of the gradient and the diagonal of the Hessian so it doesn't
        % matter).
        delta    = dinv - Dinv;
        deltaone = delta * one;
        g        = delta * y - y .* deltaone;
        dinv3    = dinv .^ 3;
        y2       = y .^ 2;
        H        = dinv3 * y2 - deltaone - 2 * y .* (dinv3 * y) + y2 .* (dinv3 * one);
        s        = -g(:) ./ abs(H(:));
        y_old    = y;

        % Use step-halving procedure to ensure progress is made
        for j=1:opts.MaxHalves
            y(:) = y_old(:) + s;
            d     = euclid(y, y) + eye(N);
            dinv  = 1 ./ d;
            delta = D - d;
            E_new = sum(sum((delta .^ 2) .* Dinv));
            if E_new < E
                break;
            else
                s = 0.5*s;
            end
        end

        % Bomb out if too many halving steps are required
        if j == opts.MaxHalves
            warning('MaxHalves exceeded. Sammon mapping may not converge...');
        end

        % Evaluate termination criterion
        if abs((E - E_new) / E) < opts.TolFun
            if display
                fprintf(1, 'Optimisation terminated - TolFun exceeded.\n');
            end
            break;
        end

        % Report progress
        E = E_new;
        if display > 1
            fprintf(1, 'epoch = %d : E = %12.10f\n', i, E * scale);
        end
    end

    % Fiddle stress to match the original Sammon paper
    E = E * scale;
end

function d = euclid(x, y)
    d = sqrt(sum(x.^2,2)*ones(1,size(y,1))+ones(size(x,1),1)*sum(y.^2,2)'-2*(x*y'));
end
