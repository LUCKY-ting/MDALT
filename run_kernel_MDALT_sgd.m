% used for datasets with non-linear decison boundary, also datasets with
% full, not sparse, data matrix
clc
clear
global trainData testData testLabel
dname = 'enron';
load(['datasets/' dname '-train.mat']);
load(['datasets/' dname '-test.mat']);

[n,d] = size(trainData);
L = size(trainLabel,2);

times = 20;  % run 20 times for calculating mean accuracy
epoch = 10;
maxIterNum = 1;

scale = 2.^(4);
miu = 0.1;
theta = 0.8;
eta = 2.^(-2.5);

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
            Y_t_size = nnz(y);
            
            for iter = 1:maxIterNum
                cur_coeff = zeros(1, L+1);
                for k = 1:L
                    if y(k) == 1
                        if pred_v(k) - pred_v(L+1) < 1 - theta
                            a_k_minus = 1;
                            a_k_plus = 0;
                        elseif  pred_v(k) - pred_v(L+1) > 1 + theta
                            a_k_minus = 0;
                            a_k_plus = 1;
                        else
                            a_k_minus = 0;
                            a_k_plus = 0;
                        end
                        cur_coeff(k) = -eta * (miu * a_k_plus - a_k_minus) / (Y_t_size * (1 - theta));
                        cur_coeff(L+1) = cur_coeff(L+1) - cur_coeff(k);
                    else
                        if pred_v(L+1) - pred_v(k) < 1 - theta
                            b_k_minus = 1;
                            b_k_plus = 0;
                        elseif  pred_v(L+1) - pred_v(k) > 1 + theta
                            b_k_minus = 0;
                            b_k_plus = 1;
                        else
                            b_k_minus = 0;
                            b_k_plus = 0;
                        end
                        cur_coeff(k) = -eta * (b_k_minus - miu * b_k_plus) / ((L - Y_t_size) * (1 - theta));
                        cur_coeff(L+1) = cur_coeff(L+1) - cur_coeff(k);
                    end
                end
                if norm(cur_coeff) == 0
                    break;
                end
                
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
fid = fopen('MDALT_OGD_result.txt','a');
fprintf(fid,'name = enron, Kernel_MDALT, runTimes = %d, scale = %g, eta = %g, miu = %g, theta = %g, epoch = %d, maxIter = %d \n', times, scale, eta, miu, theta, epoch, maxIterNum);
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


