function [C, A, Out]= solving_LR_ATSR(X, cls_num, gt, opts)   

%% Note: Multi-view subspace clustering via latent representation
%% learning and augmented tensorized self-representation
% Input:
%   X:          feature matrices
%   cls_num:    number of clusters
%   gt:         ground truth clusters
%   opts:       optional parameters
%               - maxIter: max iteration
%               - alpha, beta, etc:  hyper-parameter
%               - mu: penalty parameter
%               - epsilon: stopping tolerance
% Outout:
%   C:          clusetering results
%   A:          affinity matrix
%   Out:        other output information, e.g. metrics

%% Parameter settings
K = length(X);   % number of views
N = size(X{1},2); % sample number
% Set Discrete Fourier Transform
transform.L = @fft; transform.l = N; transform.inverseL = @ifft;

% Default
flag_debug = 0;
epsilon = 1e-7;
mu = 1e-5; 
max_mu = 1e10; 
pho_mu = 2;
maxIter = 200;


if ~exist('opts', 'var')
    opts = [];
end  
if  isfield(opts, 'maxIter');       maxIter = opts.maxIter;         end
if  isfield(opts, 'epsilon');       epsilon = opts.epsilon;         end
if  isfield(opts, 'alpha');         alpha = opts.alpha;             end
if  isfield(opts, 'beta');          beta = opts.beta;               end
if  isfield(opts, 'lambda');        lambda = opts.lambda;           end
if  isfield(opts, 'd');             d = opts.d;                     end
if  isfield(opts, 'mu');            mu = opts.mu;                   end
if  isfield(opts, 'max_mu');        max_mu = opts.max_mu;           end
if  isfield(opts, 'flag_debug');    flag_debug = opts.flag_debug;   end

%% Initialize...
Z = cell(1,K);
W = Z;
J = Z;
E = Z;
Y1 = Z;
Y2 = Z;
Y3 =Z;

L = Z;
H = Z;
P = Z;
D = Z;
G = Z;

chg1 = cell(1,K); 
chg2 = cell(1,K);
chg3 = cell(1,K);
chg4 = cell(1,K);

d1 = cell(1,K);
d2 = cell(1,K);
d3 = cell(1,K);
d4 = cell(1,K);

for k=1:K
    Z{k} = zeros(N,N); 
    W{k} = zeros(N,N);
    J{k} = zeros(N,N);
    E{k} = zeros(d,N); 
    L{k} = zeros(d,d);
    G{k} = zeros(d,d);
    D{k} = zeros(d,N);
    Y1{k} = zeros(d,N);
    Y2{k} = zeros(d,N);
    Y3{k} = zeros(d,d);
    H{k} = rand(d,N);
    P{k} = zeros(size(X{k},1),d);
end

Isconverg = 0;
iter = 0;
while(Isconverg == 0)
    if flag_debug == 1
       fprintf('----processing iter %d--------\n', iter + 1);
    end
    %% 1-------------------Update P^k------------------------------- 
    for k=1:K
        G1 = H{k};
        Q1 = X{k}';
        W1 = G1*Q1;
        [U,~,V] = svd (W1,'econ'); 
        P{k} = V*U'; 
    end

    %% 2-------------------Update H^k------------------------------- 
    for k=1:K 
        A = 2*P{k}'*P{k}/mu+eye(d,d);
        B = ((eye(N)-Z{k})*(eye(N)-Z{k})')+eye(N)*1e-10;
        C = 2*P{k}'*X{k}/mu+(L{k}*D{k}+E{k}-Y1{k}/mu)*(eye(N)-Z{k})'+D{k}-Y2{k}/mu;
        H{k} = sylvester(A, B, C);
    end

    %% 3-------------------Update Z^k-------------------------------
    for k=1:K
        HTH = H{k}'*H{k};
        tmp = HTH - H{k}'*(L{k}*D{k}+E{k}-Y1{k}/mu) +J{k}-W{k}/mu ;
        Z{k} = (HTH + eye(N))\tmp;
    end 
   
    %% 4-------------------Update L^k-------------------------------
    for k=1:K
        tmp1 = (H{k}-H{k}*Z{k}-E{k}+Y1{k}/mu)*D{k}'-Y3{k}/mu+G{k};
        L{k} = tmp1/(D{k}*D{k}'+eye(d,d));
    end

    %% 5-------------------Update D^k-------------------------------
    for k=1:K
        tmp2 = L{k}'*(H{k}-H{k}*Z{k}-E{k}+Y1{k}/mu)+Y2{k}/mu+H{k};
        D{k} = (L{k}'*L{k} + eye(d,d))\tmp2;
    end

    %% 6-------------------Update E^k-------------------------------
        C1 = [];
    for k=1:K   
        tmp1 = H{k} -H{k}*Z{k}-L{k}*D{k} + Y1{k}/mu;
        C1 = [C1; tmp1];
    end
    [Econcat] = solve_l1l2(C1,lambda/mu);
    start = 1;
    for k=1:K
        E{k} = Econcat(start:start + d - 1,:);
        start = start + d;
    end

    %% 7-------------------Update J_tensor--------------------------
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    tmp3 = Z_tensor  + W_tensor/mu;
    [J_tensor, ~] = solving_tnn(tmp3, alpha/mu, transform); 

    %% 8-------------------Update G^k-------------------------------
    for k=1:K
        tmp4 = L{k} + Y3{k}/mu;
        [U,sigma,V] = svd(tmp4,'econ');
        sigma = diag(sigma);
        svp = length(find(sigma>beta/mu));
        if svp>=1
           S1 = sigma(1:svp)-beta/mu;
           G{k} = U(:,1:svp)*diag(S1)*V(:,1:svp)';
        else
           G{k} = zeros(size(tmp4));
        end
    end
    %% 9-------------------Update auxiliary variable------------------
    for k=1:K
        J{k}  = J_tensor(:,:,k); 
        d1{k} = H{k}-H{k}*Z{k}-L{k}*D{k}-E{k};
        d2{k} = H{k} - D{k};
        d3{k} = Z{k} - J{k};
        d4{k} = L{k} - G{k};
        Y1{k} = Y1{k} + mu*(d1{k});
        Y2{k} = Y2{k} + mu*(d2{k});
        Y3{k} = Y3{k} + mu*(d4{k});
        W{k}  = W{k} + mu*(d3{k});
    end

    %% ------------------- Converge check ----------------------------
    Isconverg = 1;
    for k=1:K
       chg1{k}=norm(d1{k},inf);
        if (chg1{k}>epsilon)
            if flag_debug==1
            fprintf('norm_X   %7.10f     \n', chg1{k});
            end
            Isconverg = 0;
        end

        chg2{k}=norm(d2{k},inf);
        if (chg2{k}>epsilon)
            if flag_debug==1
            fprintf('norm_H_D %7.10f     \n', chg2{k});
            end
            Isconverg = 0;
        end

        chg3{k}=norm(d3{k},inf);
        if (chg3{k}>epsilon)
            if flag_debug==1
            fprintf('norm_Z_J %7.10f   \n', chg3{k});
            end
            Isconverg = 0;
        end

          chg4{k}=norm(d4{k},inf);
        if (chg4{k}>epsilon)
            if flag_debug==1
            fprintf('norm_L_G %7.10f   \n', chg4{k});
            end
            Isconverg = 0;
        end
    end
    
    if (iter>maxIter)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
end

%% ---------------- Clustering --------------------------------------
A = zeros(N,N);
for k=1:K
    A = A + abs(Z{k})+abs(Z{k}');
end
A = A/K;

C = SpectralClustering(A,cls_num);
[~, nmi, ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,p,r] = compute_f(gt,C);
[AR,~,~,~]=RandIndex(gt,C);
purity=compute_purity(gt,C);

%% ---------------- Record ------------------------------------------
Out.NMI = nmi;
Out.AR = AR;
Out.ACC = ACC;
Out.recall = r;
Out.precision = p;
Out.fscore = f;
Out.purity=purity;
end