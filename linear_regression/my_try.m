clc;clear;close all;
%% plot the data
data = load('ex1data1.txt');
X = data(:,1);y = data(:,2);
m = length(y); % number of training examples
figure(1)
plot(X, y, 'rx', 'MarkerSize', 10); % plot the data
ylabel('Profit in $10,000s');
xlabel('Population of city in 10,000s');
%% Gradient Descent
X = [ones(m, 1), data(:,1)]; % add a column of ones to x
theta = zeros(2,1);          % initialize fitting parameters
iterations = 1500;
alpha = 0.01;                % learn rate
% compute and display initial cost
computecost(X,y,theta)
% run gradient descent
[theta,J_history] = gradientdescent(X, y, theta, iterations, alpha);
% print theta to screeen
fprintf('Theta found by gradient descent: ')
fprintf('%f %f \n', theta(1), theta(2));
% plot the linear fit
hold on;
plot(X(:,2), X*theta, '-');
legend('Training data', 'Linear regression');
%% Visualing J(theta_0, theta_1)
fprintf('visualing J(theta_0, theta_1) ... \n');
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i,j) = computecost(X, y, t);
    end
end
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure(2)
surf(theta0_vals, theta1_vals, J_vals);
xlabel('\theta_0');ylabel('\theta_1');
figure(3)
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20));
xlabel('\theta_0');ylabel('\theta_1');
hold on
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
%% compute the cost function
function J = computecost(X, y, theta)
m = length(y);
J = 0;
J = sum((X * theta - y).^2) ./ (2*m);
end
%% algorithm of gradient descent
function [theta, J_history] = gradientdescent(X, y, theta, iterations, alpha) 
m = length(y);
J_history = zeros(iterations,1);
theta_s = theta;
for iter = 1:iterations
    theta(1) = theta(1) - alpha / m * sum((X * theta_s - y) .* X(:,1));
    theta(2) = theta(2) - alpha / m * sum((X * theta - y) .* X(:,2));
    theta_s = theta;
    J_history(iter) = computecost(X, y, theta); % save the value of cost fucntion each iteration
end
end

