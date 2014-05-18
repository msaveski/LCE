%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% DEMO
% 
% Example of how to apply the Local Collective Embeddings model on the 
% NIPS dataset to predict the most likely authors of new publications.
% 
% The data is publicly available at:
% http://www.cs.nyu.edu/~roweis/data.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% Loading the data
%
load('data/nips12raw_str602.mat', 'counts', 'apapers');
load('data/T.mat');

%
% Transforming the data in a suitable matrix form
%
Nd = size(counts, 2);
Nw = size(counts, 1);
Na = size(apapers, 1);

Xu = sparse(Nd, Na);
for i=1:Na,
	Xu(apapers{i}, i) = 1.0;
end
clear apapers;

train_time = find(sum(T(:,1:12),2));
test_time  = find(sum(T(:,13),2));

vocab = 1:Nw;
Xu_train = Xu(train_time, :);
Xs_train = counts(:, train_time)';
Xs_train = Xs_train(:, vocab);

Xu_test = Xu(test_time, :);
Xs_test = counts(:, test_time)';
Xs_test = Xs_test(:, vocab);

%
% Preprocessing the data
%
train_authors = (sum(Xu_train) > 0);
Xu_train = Xu_train(:, train_authors);
Xu_test = Xu_test(:, train_authors);

[Xs_train, Xs_test] = tfidf(Xs_train, Xs_test);

%
% Running the LCE 
%
k = 500;
alpha = 0.5;
lambda = 0.5;
epsilon = 0.001;
maxiter = 500;
verbose = true;
beta = 0.05;

% constructing the adjacency matrix
A = construct_A(Xs_train, 1, true);

fprintf('This step may take some time ... \n');
tic
[~, Hs, Hu, ObjHistory] = LCE(Xs_train,  L2_norm_row(Xu_train), A, k, alpha, beta, lambda, epsilon, maxiter, verbose);
toc
figure
plot(ObjHistory)

% Inference
W_test = Xs_test / Hs; 
W_test(W_test < 0) = 0;
LCE_ranking = W_test*Hu;

LCE_res = NDCG(LCE_ranking, Xu_test);

%
% Running the LCE without Laplacian Regularization, i.e., with beta=0
%
k = 500;
alpha = 0.5;
lambda = 0.5;
epsilon = 0.001;
maxiter = 500;
verbose = true;

fprintf('This step may take some time ... \n');
tic
[~, Hs, Hu, ObjHistory] = LCE_Beta0(Xs_train,  L2_norm_row(Xu_train), k, alpha, lambda, epsilon, maxiter, verbose);
toc
figure
plot(ObjHistory)

% Inference
W_test = Xs_test / Hs; 
W_test(W_test < 0) = 0;
LCE_Beta0_ranking = W_test*Hu;

LCE_Beta0_res = NDCG(LCE_Beta0_ranking, Xu_test);

%
% baseline: using the user profiles only
%
Ap = L2_norm_row(L2_norm_row(Xu_train)') * Xs_train;
BL_ranking = L2_norm_row(Xs_test * L2_norm_row(Ap)');
bl_res = NDCG(BL_ranking, Xu_test);

%
% Measuring NDCG
%
fprintf('LCE: %f, LCE(beta=0): %f, BL: %f \n', LCE_res, LCE_Beta0_res, bl_res);
% LCE: 0.424008, LCE(beta=0): 0.418664, BL: 0.386189 

% END
