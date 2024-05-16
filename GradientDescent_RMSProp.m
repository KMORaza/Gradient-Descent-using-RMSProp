%% Gradient Descent using Root Mean Square Propagation (RMSProp)
function [theta, costs] = gradientDescentRMSprop(X, y, theta, learning_rate, gamma, epsilon, num_iters)
    m = length(y);
    costs = zeros(num_iters, 1);
    vt = zeros(size(theta));
    for iter = 1:num_iters
        h = X * theta;
        cost = (1 / (2 * m)) * sum((h - y).^2);
        costs(iter) = cost;
        gradient = (1 / m) * X' * (h - y);
        vt = gamma * vt + (1 - gamma) * (gradient.^2);
        theta = theta - (learning_rate ./ sqrt(vt + epsilon)) .* gradient;
    end
end
m = 100;
n = 2;   
X = [ones(m, 1), rand(m, n)]; 
theta_true = [3; 2.5; 1.5]; 
y = X * theta_true + randn(m, 1); 
initial_theta = zeros(n + 1, 1); 
learning_rate = 0.01;
gamma = 0.9;
epsilon = 1e-8;
num_iters = 1000;
[theta, costs] = gradientDescentRMSprop(X, y, initial_theta, learning_rate, gamma, epsilon, num_iters);
disp('Optimized Parameters:');
disp(theta);
figure;
plot(1:num_iters, costs, '-b', 'LineWidth', 2);
xlabel('Number of Iterations');
ylabel('Cost');
title('Cost Function over Iterations');
grid on;
