clc;clear;close all;
%%
fprintf('loading data ... \n');
data = load('ex1data2.txt');
X = data(:,1:2); % input data
y = data(:,3);   % target data
m = length(y);   % the number of training data
% Print some data points
fprintf('First 10 example of the dataset are: \n');
fprintf('x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
fprintf('The programme is stopped, please press the enter to continue. \n')
pause;
[X, mu, sigma] = featurenormalize(X); % (x-mu)./sigma ~ N(0,1)
X = [ones(m,1) X]; % add another columnn to X
% Initilization 
theta = zeros(3,1);
alpha = 0.01; % learnig rate
iteration = 8500;
[theta, J_history] = gradientdescentmulti(X, y, theta, alpha, iteration);
% plot the covergency graph
figure(1);
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

price = [1 (([1650 3]-mu) ./ sigma)] * theta ;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
%% ================================Normal Equation===============================
data = csvread('ex1data2.txt');
X = data(:,1:2);
y = data(:,3);
m = length(y);
X = [ones(m,1) X];
theta = normaleqn(X, y);
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');
price = [1 1650 3] * theta ;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
%%
function [X_norm, mu, sigma] = featurenormalize(X)
X_norm = X;
mu = zeros(1, size(X, 2));   % a 1x2 matrix
sigma = zeros(1, size(X, 2));% a 1x2 matrix
mu = mean(X);  % get the mean value of X's column
sigma = std(X);% get the standard devitation of matrix X's column
X_norm = (X - repmat(mu, size(X, 1), 1)) ./ repmat(sigma, size(X,1), 1); % (x-mu)./sigma ~ N(0,1)
end
%% gradient descent muti
function [theta, J_history] = gradientdescentmulti(X, y, theta, alpha, iteration)
m = length(y); % number of training example
J_history = zeros(iteration, 1);
for iter = 1:iteration
    theta = theta - alpha / m * X' * (X * theta - y);
    J_history(iter) = computecostmulti(X, y, theta);
end
end
%% compute cost function of multi
function J = computecostmulti(X, y, theta)
m = length(y);
J = 1/(2*m) * sum(X * theta - y).^2;
end
%% 
function theta = normaleqn(X, y)
theta = pinv(X' * X) * X' * y;
end