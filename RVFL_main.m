function [Predict_Y] = RVFL_main(trainX,trainY,testX,testY,option)
N = option.N;
activation = option.activation;

s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

if ~isfield(option,'Scale')|| isempty(option.Scale)
    option.Scale=1;
end
s = option.Scale;
if ~isfield(option,'method')|| isempty(option.method)
    option.method='';
end

if strcmp(option.problem_type,'class')
    trainY_temp=GaussLabel_func(trainY,option);
else
    trainY_temp=trainY; 
end

tic
[Nsample,Nfea] = size(trainX);

W = (rand(Nfea,N)*2*s-1);
b = s*rand(1,N);
X1 = trainX*W+repmat(b,Nsample,1);

if activation == 1
    X1 = selu(X1);
elseif activation == 2
    X1 = relu(X1);
elseif activation == 3
    X1 = sigmoid(X1);
elseif activation == 4
    X1 = sin(X1);
elseif activation == 5
    X1 = hardlim(X1);        
elseif activation == 6
    X1 = tribas(X1);
elseif activation == 7
    X1 = radbas(X1);
elseif activation == 8
    X1 = sign(X1);
elseif activation == 9
    X1 = swish(X1);
end

Htrain = [trainX,X1];

Htrain = [Htrain,ones(Nsample,1)];
beta=optimize_beta2(Htrain,trainY_temp,option);

train_time=toc;

Nsample = size(testX,1);

X1 = testX*W+repmat(b,Nsample,1);

if activation == 1
    X1 = selu(X1);
elseif activation == 2
    X1 = relu(X1);
elseif activation == 3
    X1 = sigmoid(X1);
elseif activation == 4
    X1 = sin(X1);
elseif activation == 5
    X1 = hardlim(X1);        
elseif activation == 6
    X1 = tribas(X1);
elseif activation == 7
    X1 = radbas(X1);
elseif activation == 8
    X1 = sign(X1);
elseif activation == 9
    X1 = swish(X1);
end

Htest=[testX,X1,ones(Nsample,1)];

rawScore=Htest*beta;
if strcmp(option.problem_type,'class')
    rawScore_temp1 = bsxfun(@minus,rawScore,max(rawScore,[],2));
    num = exp(rawScore_temp1);
    dem = sum(num,2);
    prob_scores = bsxfun(@rdivide,num,dem);
    [max_prob,indx] = max(prob_scores,[],2);
    test_accuracy = mean(indx == testY);
    Predict_Y=indx;
else
    test_accuracy = mean(abs(rawScore-testY));
    Predict_Y=rawScore;
end
end