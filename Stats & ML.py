#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import time


# In[2]:


data_path = 'marks.txt'
data = pd.read_csv(data_path, header=None, names=['Mark1', 'Mark2', 'Admitted'])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prob(theta, x):
    return sigmoid(np.dot(x, theta))

def objective(theta, x, y):
    p = prob(theta, x)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def gradient(theta, x, y):
    return np.dot(x.T, prob(theta, x) - y)

# Gradient descent with backtracking line search
def gradient_descent_backtracking(x, y, theta, step=0.1, epsilon=0.3, beta=0.5, tolerance=1e-6, max_iterations=10000):
    start_time = time.time()
    history_theta = [theta.copy()]
    history_cost = [objective(theta, x, y)]
    
    for _ in range(max_iterations):
        grad = gradient(theta, x, y)
        obj = objective(theta, x, y)
        t = step
        while objective(theta - t * grad, x, y) > obj - epsilon * t * np.sum(grad**2):
            t *= beta
        
        theta -= t * grad
        history_theta.append(theta.copy())
        history_cost.append(objective(theta, x, y))
        
        if np.linalg.norm(t * grad) < tolerance:
            break
    time_taken = time.time() - start_time
    
    return theta, history_theta, history_cost, time_taken


# In[3]:


# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[['Mark1', 'Mark2']].values)
X_scaled_with_intercept = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

initial_theta = np.zeros(X_scaled_with_intercept.shape[1])

final_theta, history_theta, history_cost, time_taken = gradient_descent_backtracking(
    X_scaled_with_intercept, data['Admitted'].values, initial_theta
)


# In[4]:


# Final theta, cost, number of iterations, and time taken 
history_theta[-1], history_cost[-1], len(history_theta)-1, time_taken


# In[5]:


# 3D plot of the weight vectors vs cost
weights = np.array(history_theta)
costs = np.array(history_cost)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(weights[:, 0], weights[:, 1], costs, label='Gradient Descent Path', color='r', marker='o')
ax.set_xlabel('Intercept')
ax.set_ylabel('Weight 1')
ax.set_zlabel('Cost')
ax.set_title('3D plot of Weight Vectors vs. Cost')
ax.legend()
plt.show()

# 2D contour plot of the cost function
weight_1_range = np.linspace(weights[:, 1].min() - 1, weights[:, 1].max() + 1, 100)
weight_2_range = np.linspace(weights[:, 2].min() - 1, weights[:, 2].max() + 1, 100)
weight_1_grid, weight_2_grid = np.meshgrid(weight_1_range, weight_2_range)
cost_grid = np.zeros_like(weight_1_grid)
for i in range(weight_1_grid.shape[0]):
    for j in range(weight_1_grid.shape[1]):
        cost_grid[i, j] = objective(
            np.array([final_theta[0], weight_1_grid[i, j], weight_2_grid[i, j]]),
            X_scaled_with_intercept,
            data['Admitted'].values
        )
plt.figure(figsize=(10, 6))
contour = plt.contour(weight_1_grid, weight_2_grid, cost_grid, levels=np.logspace(-2, 3, 20), cmap=plt.cm.jet)
plt.clabel(contour, inline=1, fontsize=10)
plt.plot(weights[:, 1], weights[:, 2], label='Path of Gradient Descent', color='r', marker='x')
plt.plot(final_theta[1], final_theta[2], 'ro')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('Contour plot of Cost Function with Gradient Descent Path')
plt.legend()
plt.show()


# In[6]:


def accuracy(theta, x, y):
    predictions = predict(theta, x)
    return np.mean(predictions == y)

def predict(theta, x):
    probabilities = prob(theta, x)
    return probabilities >= 0.5

model_accuracy = accuracy(final_theta, X_scaled_with_intercept, data['Admitted'].values)
print(f'Accuracy of the logistic regression model: {model_accuracy * 100:.2f}%')

admitted = data[data['Admitted'] == 1]
not_admitted = data[data['Admitted'] == 0]

# Plot the decision boundary and the data points with colors
plt.figure(figsize=(10, 6))

colors = ['red' if label == 0 else 'blue' for label in data['Admitted']]
plt.scatter(admitted['Mark1'], admitted['Mark2'], c='blue', label='Admitted')
plt.scatter(not_admitted['Mark1'], not_admitted['Mark2'], c='red', label='Not Admitted')

plot_x = np.array([min(X_scaled_with_intercept[:, 1]) , max(X_scaled_with_intercept[:, 1]) ])
plot_y = (-1 / final_theta[2]) * (final_theta[1] * plot_x + final_theta[0])

plot_y = scaler.inverse_transform(np.vstack([np.zeros_like(plot_y), plot_y]).T)[:, 1]
plot_x = scaler.inverse_transform(np.vstack([plot_x, np.zeros_like(plot_x)]).T)[:, 0]

plt.plot(plot_x, plot_y, label='Decision Boundary', color='green')

plt.xlabel('Mark 1')
plt.ylabel('Mark 2')
plt.legend()
plt.show()


# In[7]:


# Plot the decision boundary and the data points with color on the background
plt.figure(figsize=(10, 6))

x_min, x_max = data['Mark1'].min() - 1, data['Mark1'].max() + 1
y_min, y_max = data['Mark2'].min() - 1, data['Mark2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
grid_scaled_with_intercept = np.hstack((np.ones((grid_scaled.shape[0], 1)), grid_scaled))
Z = 1-predict(final_theta, grid_scaled_with_intercept)

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.bwr)

admitted = data[data['Admitted'] == 1]
not_admitted = data[data['Admitted'] == 0]

colors = ['red' if label == 0 else 'blue' for label in data['Admitted']]
plt.scatter(admitted['Mark1'], admitted['Mark2'], c='blue', label='Admitted')
plt.scatter(not_admitted['Mark1'], not_admitted['Mark2'], c='red', label='Not Admitted')

plt.plot(plot_x, plot_y, label='Decision Boundary', color='green')

plt.xlabel('Mark 1')
plt.ylabel('Mark 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend()
plt.show()


# In[8]:


# Hessian function for logistic regression
def hessian(theta, x):
    p = prob(theta, x)
    W = np.diag(p * (1 - p))
    return np.dot(x.T, np.dot(W, x))

# Newton's method for logistic regression
def newtons_method(x, y, theta, tolerance=1e-6, max_iterations=1000):
    start_time = time.time()
    history_theta = [theta.copy()]
    history_cost = [objective(theta, x, y)]
    
    for _ in range(max_iterations):
        grad = gradient(theta, x, y)
        H = hessian(theta, x)
        theta_update = np.linalg.inv(H).dot(grad)
        
        theta -= theta_update
        history_theta.append(theta.copy())
        history_cost.append(objective(theta, x, y))

        if np.linalg.norm(theta_update) < tolerance:
            break
    time_taken = time.time() - start_time
    return theta, history_theta, history_cost, time_taken


# In[9]:


initial_theta = np.zeros(X_scaled_with_intercept.shape[1])
# Run Newton's method
final_theta_newton, history_theta_newton, history_cost_newton, time_taken_newton = newtons_method(
    X_scaled_with_intercept, data['Admitted'].values, initial_theta
)


# In[10]:


# Final theta, cost, number of iterations, and time taken 
final_theta_newton, history_cost_newton[-1], len(history_theta_newton)-1, time_taken_newton 


# In[11]:


# 3D plot of the weight vectors vs cost
weights_newton = np.array(history_theta_newton)
costs = np.array(history_cost_newton)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(weights_newton[:, 0], weights_newton[:, 1], costs, label='Gradient Descent Path', color='r', marker='o')
ax.set_xlabel('Intercept')
ax.set_ylabel('Weight 1')
ax.set_zlabel('Cost')
ax.set_title('3D plot of Weight Vectors vs. Cost')
ax.legend()
plt.show()

# Adjust the range of the grid to show a larger region around the final weights
extend_factor = 2
weight_1_range_ext = np.linspace(weights[:, 1].min() - extend_factor, weights[:, 1].max() + extend_factor, 100)
weight_2_range_ext = np.linspace(weights[:, 2].min() - extend_factor, weights[:, 2].max() + extend_factor, 100)
weight_1_grid_ext, weight_2_grid_ext = np.meshgrid(weight_1_range_ext, weight_2_range_ext)

cost_grid_ext = np.zeros_like(weight_1_grid_ext)
for i in range(weight_1_grid_ext.shape[0]):
    for j in range(weight_1_grid_ext.shape[1]):
        cost_grid_ext[i, j] = objective(
            np.array([final_theta_newton[0], weight_1_grid_ext[i, j], weight_2_grid_ext[i, j]]),
            X_scaled_with_intercept,
            data['Admitted'].values,
        )

# 2D contour plot of the cost function on the extended grid
plt.figure(figsize=(12, 8))
contour_ext = plt.contour(weight_1_grid_ext, weight_2_grid_ext, cost_grid_ext, levels=np.logspace(-2, 3, 20), cmap=plt.cm.jet)
plt.clabel(contour_ext, inline=1, fontsize=10)

plt.plot(weights_newton[:, 1], weights_newton[:, 2], label='Path of Newton\'s Method', color='r', marker='x')

plt.plot(final_theta_newton[1], final_theta_newton[2], 'bo') 

plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('Extended Contour plot of Cost Function with Newton\'s Method Path')
plt.legend()
plt.show()


# In[12]:


def accuracy(theta, x, y):
    predictions = predict(theta, x)
    return np.mean(predictions == y)

def predict(theta, x):
    probabilities = prob(theta, x)
    return probabilities >= 0.5

model_accuracy = accuracy(final_theta_newton, X_scaled_with_intercept, data['Admitted'].values)
print(f'Accuracy of the logistic regression model: {model_accuracy * 100:.2f}%')

admitted = data[data['Admitted'] == 1]
not_admitted = data[data['Admitted'] == 0]

# Plot the decision boundary and the data points with colors
plt.figure(figsize=(10, 6))

colors = ['red' if label == 0 else 'blue' for label in data['Admitted']]
plt.scatter(admitted['Mark1'], admitted['Mark2'], c='blue', label='Admitted')
plt.scatter(not_admitted['Mark1'], not_admitted['Mark2'], c='red', label='Not Admitted')

plot_x = np.array([min(X_scaled_with_intercept[:, 1]) , max(X_scaled_with_intercept[:, 1]) ])
plot_y = (-1 / final_theta_newton[2]) * (final_theta_newton[1] * plot_x + final_theta_newton[0])

plot_y = scaler.inverse_transform(np.vstack([np.zeros_like(plot_y), plot_y]).T)[:, 1]
plot_x = scaler.inverse_transform(np.vstack([plot_x, np.zeros_like(plot_x)]).T)[:, 0]

plt.plot(plot_x, plot_y, label='Decision Boundary', color='green')

plt.xlabel('Mark 1 Score')
plt.ylabel('Mark 2 Score')
plt.legend()
plt.show()


# In[13]:


# Plot the decision boundary and the data points with color on the background
plt.figure(figsize=(10, 6))

x_min, x_max = data['Mark1'].min() - 1, data['Mark1'].max() + 1
y_min, y_max = data['Mark2'].min() - 1, data['Mark2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
grid_scaled_with_intercept = np.hstack((np.ones((grid_scaled.shape[0], 1)), grid_scaled))
Z = 1-predict(final_theta, grid_scaled_with_intercept)

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.bwr)

admitted = data[data['Admitted'] == 1]
not_admitted = data[data['Admitted'] == 0]

colors = ['red' if label == 0 else 'blue' for label in data['Admitted']]
plt.scatter(admitted['Mark1'], admitted['Mark2'], c='blue', label='Admitted')
plt.scatter(not_admitted['Mark1'], not_admitted['Mark2'], c='red', label='Not Admitted')

plt.plot(plot_x, plot_y, label='Decision Boundary', color='green')

plt.xlabel('Mark 1 Score')
plt.ylabel('Mark 2 Score')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend()
plt.show()


# In[14]:


# Newton's method with backtracking line search
def newtons_method_with_backtracking(x, y, theta, tolerance=1e-6, max_iterations=100, epsilon=0.7, beta=0.8):
    start_time = time.time()
    history_theta = [theta.copy()]
    history_cost = [objective(theta, x, y)]
    
    for _ in range(max_iterations):
        grad = gradient(theta, x, y)
        H = hessian(theta, x)
        newton_step = np.linalg.inv(H).dot(grad)
        
        # Backtracking line search
        t = 1
        while True:
            theta_new = theta - t * newton_step
            if objective(theta_new, x, y) < (objective(theta, x, y) - epsilon * t * np.dot(grad, newton_step)):
                break  
            t *= beta  
        
        theta = theta_new
        history_theta.append(theta.copy())
        history_cost.append(objective(theta, x, y))
        
        if np.linalg.norm(newton_step) * t < tolerance:
            break
    time_taken = time.time() - start_time
    return theta, history_theta, history_cost, time_taken


# In[15]:


initial_theta = np.zeros(X_scaled_with_intercept.shape[1])
# Run Newton's method with backtracking line search
final_theta_newton_bt, history_theta_newton_bt, history_cost_newton_bt, time_taken_newton_bt = newtons_method_with_backtracking(
    X_scaled_with_intercept, data['Admitted'].values, initial_theta
)


# In[16]:


# Final theta, cost, number of iterations, time taken 
final_theta_newton_bt, history_cost_newton_bt[-1], len(history_theta_newton_bt) - 1, time_taken_newton_bt


# # Conclusion
# ### Part (a)
# 
# Implemented a gradient descent optimizer with backtracking line search for logistic regression. The optimizer successfully minimized the cost function using a step size adjustment strategy to ensure sufficient decrease at each iteration.
# 
# ### Part (b)
# The gradient descent optimizer with backtracking line search converged to a solution close to the one obtained using the fmin_tnc method.The accuracy is 89%, matching the pdf example. Both the accuracy the decision boundary plots confirm the effectiveness of the custom optimizer.
# 
# ### Part (c)
# Plotted the sequence of weight vectors during the optimization process. The plots showed how the weights updated at each step and how the optimizer converged towards the minimum of the cost function.
# 
# ### Part (d)
# Implemented Newton's method using the exact Hessian derived in the lectures. Newton's method converged in eight iterations for this problem, indicating that it was very efficient given the specific logistic regression problem and data provided.
# 
# Convergence to the correct solution: Both methods converged to similar and appropriate solutions.
# 
# Number of steps taken to converge: Newton's method converged in fewer steps (8 steps) than gradient descent (with various step sizes and $\epsilon$, e.g. 189 steps when initial step size is 0.1), which is expected given its use of second-order information. 
# 
# Overall time used for convergence: Newton's method is faster (around 0.002s) than the gradient descent (around 0.020s) in this problem, but I think this might not generalize to larger problems due to the cost of computing and inverting the Hessian.
# 
# Adding backtracking line search: For this problem, backtracking line search would probably not help much and could potentially slow down the convergence of Newton's method, which already rapidly finds the optimum due to the Hessian providing a good step direction and size.In our implementation, for small $\epsilon$, there is basically no difference with the original Newton's method. And for lagre $\epsilon$, the iterations and time needed can increase (26 steps and 0.017s). Therefore, I think we don't need to add backtracking line search to Newton's method. 
# 
# #### In conclusion, both the custom gradient descent and Newton's method proved to be effective optimization strategies for logistic regression, with Newton's method showing superior efficiency in this case.
