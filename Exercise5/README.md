# Exercise 5 (week 6: Regularized Linear Regression and Bias v.s.Variance)

We implement regularized linear regression and use it to study models with different bias-variance properties.

Starting with a simple linear regression fit:
<!--- ![f1](F1_1d_fit.png) --->
<img src="F1_1d_fit.png" width="80%">
We observe in the learning curve (below) that the cost function is relatively high both for the training and cross validation sets. The don't decrease with more data (training samples). This is an indication of bias.
<!--- ![f2](F2_learning_curve_lin.png) --->
<img src="F2_learning_curve_lin.png" width="80%">
Next step is to increase the number of features by using a polynomial fit (8-dimensional). We start without regularization. 
<img src="F3_8d_polyfit_L0.png" width="80%">
This fits the training data well, but performs poorly on cross validation. We observe in the learning curve (below) that the cost function for the training set is low, but for the validation set it remains high also for larger training sets. This is an indication of high variance (overfitting)
<img src="F4_learning_curve_poly_L0.png" width="80%">
In order to find a suitable regularization factor, we calculate the validation curve (running optimizations and tests for different lambdas). 
<img src="F5_validation_curve.png" width="80%">
Lambda = 3 appears to be a good choice. This yields:
<img src="F3_8d_polyfit.png" width="80%">
With our optimal learning curve:
<img src="F4_learning_curve_poly.png" width="80%">

