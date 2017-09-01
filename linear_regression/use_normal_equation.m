data = load('ex1data2.txt');
y = data(:,3);
X = [ones(length(y),1) data(:,1:2)];
theta = inv(X' * X) * X' * y;
fprintf('Theta computed from the equation is: \n');
fprintf('%f \n', theta);