function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

%X_norm = (X - mu)/sigma
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);%@minus代表的是减法

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);%@rdivide代表的是除法

% ============================================================

end
