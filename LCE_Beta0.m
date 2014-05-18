function [W, Hs, Hu, ObjHistory] = LCE_Beta0(Xs, Xu, k, alpha, lambda, epsilon, maxiter, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Local Collective Embeddings (LCE) WITHOUT Laplacian regularization, i.e.,
% the parameter $\beta=0$.
% 
% Optimizes the following objective function using multiplicative update rules:
% $ min: alpha*||Xs - W*Hs||^2 + ||Xu - W*Hu||^2 + lambda*W + lambda*Hs + lambda*Hu $
% $ s.t. \W \geq 0, \Hs \geq 0, \Hu \geq 0 $
% 
% 
% 
% Hyper-parameters:
% 
%     $k$: number of topics. Controls the complexity of the model.
% 
%     $\alpha \in [0, 1]$ controls the importance of each factorization.
%         Setting $\alpha = 0.5$ gives equal importance to both factorizations, while 
%         values of $\alpha >0.5$ (or $\alpha < 0.5$) give more importance to the 
%         factorization of $\Xs$ (or $\Xu$).
%         Default: 0.5
% 
%     $\lambda$: hyper-paramter controlling the Thikonov Regularization.
%         Enforces smooth solutions and avoids over-fitting.
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

% constants
beta = 1.0 - alpha;
trXstXs = tr(Xs, Xs);
trXutXu = tr(Xu, Xu);

% values for the 1st iteration
WtW = W' * W;
WtXs = W' * Xs;
WtXu = W' * Xu;
WtWHs = WtW * Hs;
WtWHu = WtW * Hu;

% iteration counters
itNum = 1;
delta = 2 * epsilon;

% main loop
while((delta > epsilon) && (itNum <= maxiter)),
	% ================ UPDATE Hs & Hu  ================
	Hs = Hs .* ((alpha * WtXs) ./ max((alpha * WtWHs + lambda * Hs), 1e-10));
	Hu = Hu .* ((beta * WtXu)  ./ max((beta  * WtWHu + lambda * Hu), 1e-10));
	
	% =================== UPDATE W ====================
	W = W .* ((alpha*Xs*Hs' + beta*Xu*Hu') ./ max((alpha*W*Hs*Hs' + beta*W*Hu*Hu' + lambda*W), 1e-10));
	
	% ======= Calculating the objective function ======
	WtW = W' * W;
	WtXs = W' * Xs;
	WtXu = W' * Xu;
	WtWHs = WtW * Hs;
	WtWHu = WtW * Hu;
	
  tr1 = alpha  * (trXstXs - 2*tr(Hs, WtXs) + tr(Hs, WtWHs));
  tr2 = beta   * (trXutXu - 2*tr(Hu, WtXu) + tr(Hu, WtWHu));
  tr3 = lambda * (trace(WtW) + tr(Hs, Hs) + tr(Hu, Hu));
  Obj = tr1 + tr2 + tr3;
	
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
