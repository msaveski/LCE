function [W, Hs, Hu, ObjHistory] = LCE(Xs, Xu, A, k, alpha, beta, lambda, epsilon, maxiter, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Local Collective Embeddings (LCE)
% 
% Optimizes the following objective function using multiplicative update rules:
% $ min: alpha*||Xs - W*Hs||^2 + ||Xu - W*Hu||^2 + beta*trace(W'*L*W) + lambda*W + lambda*Hs + lambda*Hu $
% $ s.t. \W \geq 0, \Hs \geq 0, \Hu \geq 0 $
% 
% 
% 
% Parameters:
%     $A$: Adjacency matrix.
%          $A_ij$ corresponds to the similarity between documents i and documents j in Xs or Xu.
%          A is assumed to be symmetric (knn graphs are not necessarily symmetric).
%        
%     $k$: number of topics. Controls the complexity of the model.
% 
%     $\alpha \in [0, 1]$ controls the importance of each factorization.
%         Setting $\alpha = 0.5$ gives equal importance to both factorizations, while 
%         values of $\alpha >0.5$ (or $\alpha < 0.5$) give more importance to the 
%         factorization of $\Xs$ (or $\Xu$).
%         Default: 0.5
%
%     $\beta$: Controls the influence of the Laplacian regularization.
% 
%     $\lambda$: hyper-paramter controlling the Thikonov Regularization.
%         Enforces smooth solutions and avoids over fitting.
%         Default: 0.5
% 
%     $\epsilon$: Threshold controlling the number of iterations.
%         If the objective function decreases less then $\epsilon$ form one iteration
%         to another, the optimization procedure is stopped.
% 
%     $maxiter$: Maximum number of iterations. 
% 
%     $verbose$: True|False. If set to True prints the value of the objective function 
%         in each iteration.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fix seed for reproducible experiments
rand('seed', 354);

% initialization
n = size(Xs, 1);
v1 = size(Xs, 2);
v2 = size(Xu, 2);

% randomly initialize W, Hu, Hs.
W  = abs(rand(n, k));
Hs = abs(rand(k, v1));
Hu = abs(rand(k, v2));

D = sparse(diag(sum(A)));

% constants
gamma = 1 - alpha;
trXstXs = tr(Xs, Xs);
trXutXu = tr(Xu, Xu);

% values for the 1st iteration
WtW = W' * W;
WtXs = W' * Xs;
WtXu = W' * Xu;
WtWHs = WtW * Hs;
WtWHu = WtW * Hu;
DW = D * W;
AW = A * W;

% iteration counters
itNum = 1;
delta = 2 * epsilon;

% main loop
while(delta > epsilon) &&( (itNum <= maxiter)),
  % ================ UPDATE H ================
  Hs = Hs .* ((alpha * WtXs) ./ max((alpha * WtWHs + lambda * Hs), 1e-10));
  Hu = Hu .* ((gamma * WtXu) ./ max((gamma * WtWHu + lambda * Hu), 1e-10));

  % ================ UPDATE W ================
  W = W .* ((alpha*Xs*Hs' + gamma*Xu*Hu' + beta*AW) ./ max((alpha*(W*(Hs*Hs')) + gamma*(W*(Hu*Hu')) + beta*DW + lambda*W), 1e-10));

  % === Calculating the objective function ===
  WtW = W' * W;
  WtXs = W' * Xs;
  WtXu = W' * Xu;
  WtWHs = WtW * Hs;
  WtWHu = WtW * Hu;
  DW = D * W;
  AW = A * W;

  tr1 = alpha  * (trXstXs - 2*tr(Hs, WtXs) + tr(Hs, WtWHs));
  tr2 = gamma  * (trXutXu - 2*tr(Hu, WtXu) + tr(Hu, WtWHu));
  tr3 = beta   * (tr(W, DW) - tr(W, AW));
  tr4 = lambda * (trace(WtW) + tr(Hs, Hs) + tr(Hu, Hu));
  Obj = tr1 + tr2 + tr3 + tr4;
  ObjHistory(itNum) = Obj;

  if itNum ~= 1
    delta = abs(ObjHistory(itNum) - ObjHistory(itNum - 1));
    if verbose, fprintf('Iteration: %d \t Objective: %f \t Delta: %f \n', itNum, Obj, delta); end
  else
    if verbose, fprintf('Iteration: %d \t Objective: %f \n', itNum, Obj); end
  end

  itNum = itNum + 1;
end

% Efficient calculation of traces
function [trAB] = tr(A, B)
	trAB = sum(sum(A.*B));
end

end
% END
