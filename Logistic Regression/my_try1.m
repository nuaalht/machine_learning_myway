clc,clear,close all;
%% load data
data = load('ex2data1.txt');
X = data(:,1:2); y = data(:,3);
%% visualize the grades of the students
plotdata(X,y);
%% compute cost and gradient
[m,n] = size(X);
X = [ones(m,1) X]; % extend X,add a column that all 1
initial_theta = zeros(n+1,1);
%% use fminunc fuction to calculate the minimum value of costfunction 
% set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
% set the GraObj on means both return the cost and gradient
% set MaxIter 400 means the fminunc will run at most 400 iterations
% before it terminates
% Run fminunc to obtain the optimal theta
% This fuction will return theta and cost
[theta, cost] = fminunc(@(t)(costfunction(t, X, y)), initial_theta, options);
%% Plot Boundary
plotdecisionboundary(theta, X, y);
%% sigmode function
function g = Sigmoid(z)
g = 1./(1+exp(-z));
end
%% cost fuction of logistic regression
function [J, grad] = costfunction(theta, X, y)
m = length(y);
grad = zeros(size(theta));
J = 0;
J = -1* sum(y .* log( Sigmoid(X*theta)) + (1 - y) .* log(1-Sigmoid(X*theta))) /m;
grad = (X' * (Sigmoid(X*theta) - y)) /m; 
end
%% plot decision boundary function
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
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
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
%% visualize function
function plotdata(X,y)
pos = find( y== 1); neg = find(y == 0);
plot(X(pos,1), X(pos,2), '+', 'LineWidth', 2, 'MarkerSize', 7)
hold on;
plot(X(neg,1), X(neg,2), 'o', 'MarkerFaceColor', 'y','MarkerSize', 7)
xlabel('Exam 1 score'); ylabel('Exam 2 score');
legend('Admitted', 'Not Admitted');
end
%% function mapfeature
function out = mapfeature(X1, X2)
degree = 6; 
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end
end