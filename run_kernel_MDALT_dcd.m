% used for datasets with non-linear decison boundary, also datasets with
% full, not sparse, data matrix
clc
clear
global trainData testData testLabel
dname = 'enron';
load(['datasets/' dname '-train.mat']);
load(['datasets/' dname '-test.mat']);

epoch = 10;
times = 20;  % run 20 times for calculating mean accuracy

[n,d] = size(trainData);
L = size(trainLabel,2);

scale = 2.^(5);
delta = 0.1;
theta = 0.7;
eta = 2.^(-1.5);
maxIterNum = 1;


sr = RandStream.create('mt19937ar','Seed',1);
RandStream.setGlobalStream(sr);
test_macro_F1_score = zeros(times,1);
test_micro_F1_score = zeros(times,1);
hammingLoss = zeros(times,1);
rankingLoss = zeros(times,1);
subsetAccuracy = zeros(times,1);
oneError = zeros(times,1);
precision = zeros(times,1);
recall = zeros(times,1);
F1score = zeros(times,1);
testTime = zeros(times,1);

kernelMatrix = importdata(['kernel_matrix/' dname '/kernelMatrix_scale_' num2str(scale) '.mat']);

tStart = tic;
for run = 1:times
    coff = zeros(n, L+1);
    SVsIdx = zeros(1, n);
    SVsNum = 0;
    for o = 1:epoch
        index = randperm(n);
        for i=1:n
            j = index(i);
            x = trainData(j,:)';
            y = trainLabel(j,:);
            t = (o - 1)*n + i;
            
            if t == 1
                pred_v = zeros(1,L+1);
                km = [];
            else
                km = kernelMatrix(SVsIdx(1:SVsNum),j);
                pred_v = km'* coff(1:SVsNum,:);
            end
            
            % pred_y = pred_v(1:L) > pred_v(L+1); %online prediction
            R_Idx = find(y);
            nR_Idx = find(y==0);
            Y_t_size = nnz(y);
            nY_t_size = L - Y_t_size;
            x_norm = 1;
            A1 = [eye(Y_t_size)  -1*eye(Y_t_size); -1*eye(Y_t_size)  eye(Y_t_size)];
            A2 = [eye(nY_t_size)  -1*eye(nY_t_size); -1*eye(nY_t_size)  eye(nY_t_size)];
            A3 = zeros(2*Y_t_size,2*nY_t_size);
            Q = [A1 A3; A3' A2] + [ones(Y_t_size,1); -1*ones(Y_t_size,1); -1*ones(nY_t_size,1); ones(nY_t_size,1)] * [ones(Y_t_size,1); -1*ones(Y_t_size,1); -1*ones(nY_t_size,1); ones(nY_t_size,1)]';
            B1 = [eye(Y_t_size); -eye(Y_t_size); zeros(2*nY_t_size,Y_t_size)];
            B2 = [zeros(2*Y_t_size,nY_t_size); eye(nY_t_size); -eye(nY_t_size)];
            fst_q = [B1 B2];
            kappa = eta/(1 - theta) * [repmat(1/Y_t_size,1,Y_t_size) repmat(delta/Y_t_size,1,Y_t_size) repmat(1/nY_t_size,1,nY_t_size) repmat(delta/nY_t_size,1,nY_t_size)];
            
            
            for iter = 1:maxIterNum
                cur_coeff = zeros(1, L+1);
                q_v = fst_q * [pred_v(R_Idx(1:end)) - (pred_v(L+1) + 1) * ones(1,Y_t_size)  (pred_v(L+1) - 1) * ones(1,nY_t_size) - pred_v(nR_Idx(1:end))]' + theta * ones(2*L,1);
                gamma = zeros(2*L,1);
                func_value = 0;
                ratio = inf;
                while abs(ratio) > 1e-3
                    old_func_value = func_value;
                    pailie = randperm(2*L);
                    for k = 1:2*L
                        h = pailie(k);  %updating the h-th component of gamma
                        q_h = q_v(h);
                        gradient = x_norm * Q(h,:)* gamma + q_h;
                        kappa_h = kappa(h);
                        if gradient ~= 0
                            uncut = gamma(h) - gradient / (2 * x_norm);
                            gamma(h) = min(max(0, uncut), kappa_h);
                            %func_value = (gamma' * Q * gamma) * x_norm / 2 + q_v' * gamma
                        end
                    end
                    func_value = (gamma' * Q * gamma) * x_norm / 2 + q_v' * gamma;
                    ratio = (func_value - old_func_value)/old_func_value;
                end
                cur_coeff(R_Idx) = (gamma(1:Y_t_size) - gamma(Y_t_size+1 : 2*Y_t_size))';
                cur_coeff(nR_Idx) = - (gamma(1+2*Y_t_size: nY_t_size+2*Y_t_size) - gamma(1+2*Y_t_size + nY_t_size:end))';
                cur_coeff(L+1) = - (sum(gamma(1:Y_t_size)) - sum(gamma(1+Y_t_size:Y_t_size+L)) + sum(gamma(Y_t_size+L+1:end)))';
                clear gamma
                
                if o == 1
                    if iter == 1
                        SVsNum = SVsNum + 1;
                        SVsIdx(SVsNum) = j;
                        curId = SVsNum;
                        km = [km; 1];
                    end
                else
                    if iter == 1
                        id = find(SVsIdx(1:SVsNum) == j,1);
                        if ~isempty(id)
                            curId = id;
                        else
                            SVsNum = SVsNum + 1;
                            SVsIdx(SVsNum) = j;
                            curId = SVsNum;
                            km = [km; 1];
                        end
                    end
                end
                coff(curId, :) = coff(curId, :) + cur_coeff;
                %re-compute the predicted value for all labels
                pred_v = pred_v + km(curId).* cur_coeff;
            end
        end
    end
    
    tic
    %-------------evaluate model performance on test data-------------------------
    [test_macro_F1_score(run), test_micro_F1_score(run), hammingLoss(run), subsetAccuracy(run),  ...
        precision(run), recall(run), F1score(run), rankingLoss(run), oneError(run)] = testEvaluate_kernel_efficient(SVsIdx(1:SVsNum),coff,SVsNum,scale);
    clear coff SVsIdx
    testTime(run) = toc;
end
totalTime = toc(tStart);
avgTestTime = mean(testTime);
avgTrainTime = (totalTime - sum(testTime))/times;

clear kernelMatrix
%-------------output result to file----------------------------------------
fid = fopen('MDALT_DCD_result.txt','a');
fprintf(fid,'name = enron, Kernel_MDALT_dcd, runTimes = %d, scale = %g, eta = %g, delta = %g, theta = %g \n', times, scale, eta, delta, theta);
fprintf(fid,'epoch = %d, maxIter = %d \n', epoch, maxIterNum);
fprintf(fid,'precision +std,  recall +std,  F1score +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,\n', mean(precision), std(precision), mean(recall), std(recall), mean(F1score), std(F1score));
fprintf(fid,'macro_F1score +std, micro_F1score +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f \n ', mean(test_macro_F1_score), std(test_macro_F1_score), mean(test_micro_F1_score), std(test_micro_F1_score));
fprintf(fid,'hammingloss +std, subsetAccuracy +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f,\n ', mean(hammingLoss), std(hammingLoss), mean(subsetAccuracy), std(subsetAccuracy));
fprintf(fid,'rankingLoss +std, oneErr +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f \n ', mean(rankingLoss), std(rankingLoss), mean(oneError), std(oneError));
fprintf(fid,'training time [s], testing time [s]\n');
fprintf(fid,'%.4f, %.4f \n\n', avgTrainTime, avgTestTime);
fclose(fid);


