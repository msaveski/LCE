function [ A ] = construct_A(X, k, binary)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Constructs an adjacency matrix based on the nearest neighbor graph
%
% Parameters:
%   X: data matrix where every point is a row
%   
%   k: number of nearest neighbors to be included in the graph
%   
%   binary: boolean, whether to include boolean values or the cosine
%       similarity.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = size(X, 1);

% computing cosine similarity between all pairs of points
X = L2_norm_row(X);
S = X * X';

% find the k nearest neighbors
[vals, inds] = sort(S, 2, 'descend');
vals = vals(:, 2:(k+1));
inds = inds(:, 2:(k+1));

R = [repmat(1:n, 1, k)', inds(:), vals(:)];
A = sparse(R(:,1), R(:,2), R(:,3), n, n);
A = max(A, A');

if binary,
    A(A>0) = 1;
end

end