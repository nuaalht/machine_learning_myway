clc,clear,close all;
%% load data
data = load('ex2data2.txt');
X = data(:,1:2); y = data(:,3);
%% visualize the data
plotdata(X,y);
%%
% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapfeature(X(:,1), X(:,2)); % [ones(length(X),1), X1, X2, X1.^2, X1.*X2, X2.^2, X1.^3, X1.^2.*X2, X1.*X2.^2, X2.^3, ...]

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1); % zeros(28, 1)

% Set regularization parameter lambda to 1
lambda = 1;
%%
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costfunctionReg(initial_theta, X, y, lambda);
%%
% set options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costfunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotdecisionboundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
%% visualize data function
function plotdata(X,y)
pos = find(y==1); neg = find(y==0);
plot(X(pos,1), X(pos,2), 'k+', 'MarkerSize', 7, 'LineWidth', 2);
hold on;
plot(X(neg,1), X(neg,2), 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'y');
legend('y=1','y=0');
xlabel('Microchip Test 1');ylabel('Microchip Test 2')
hold off;
end
%% function mapfeature
function out = mapfeature(X1, X2)
degree = 6; 
out = ones(size(X1(:,1))); 
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)) .* (X2.^j);
    end
end
end
%% costfunctionReg
function [J, grad] = costfunctionReg(theta, X ,y, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));

theta_1 = [0;theta(2:end)];    % 先把theta(1)拿掉，不参与正则化
J= -1 * sum( y .* log( sigmoid(X*theta) ) + (1 - y ) .* log( (1 - sigmoid(X*theta)) ) ) / m  + lambda/(2*m) * (theta_1' * theta_1);
grad = ( X' * (sigmoid(X*theta) - y ) )/ m + lambda/m * theta_1 ;
end
%% plotdecisonboundary
function plotdecisionboundary(theta, X, y)
plotdata(X(:,2:3),y);
hold on;

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1)); 
    % theta(1)*x1 + theta(2)*x2 + theta(3)*x3 = y 
    % set y=0, then x3 = -1./theta(3)*(theta(1)*x1 + theta(2)*x2)
    % where x1=1, then x3 = -1./theta(3)*(theta(1) + theta(2)*x2)
                                                            
    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50); % get 50 points betweenn -1 and 1.5
    v = linspace(-1, 1.5, 50); % get 50 points betweenn -1 and 1.5

    z = zeros(length(u), length(v)); % z = zeros(50, 50)
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapfeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off
end
%% function predit
function p = predict(theta, X)
m = size(X, 1); % number of training examples
p = zeros(m, 1); % rteturn the following vatriables correctly
k = find(sigmoid(X * theta) >= 0.5);
p(k) = 1;
end
%% function sigmoid
function g = sigmoid(z)
g = 1./(1 + exp(-z));
end