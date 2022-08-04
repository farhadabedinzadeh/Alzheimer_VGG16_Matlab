clear;close;clc

%Kaggle Dataset
% https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images?select=Alzheimer_s+Dataset

%% Dataset
imds=imageDatastore('Data', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% Train, Test and Validation Dataset

[imdsTrain, imdsValid,imdsTest]=splitEachLabel(imds,0.75,0.15,0.10);
%% Unbalance confirmation
labelCount = countEachLabel(imdsTrain)
% histogram(imdsTrain.Labels)
labels=imdsTrain.Labels;
[G,classes] = findgroups(labels);
numObservations = splitapply(@numel,labels,G);

%% SMOTE
desiredNumObservationsPerClass = max(numObservations);

files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},imdsTrain.Files,G);
files = vertcat(files{:});
labels=[];info=strfind(files,'\');
for i=1:numel(files)
    idx=info{i};
    dirName=files{i};
    targetStr=dirName(idx(end-1)+1:idx(end)-1);
    targetStr2=cellstr(targetStr);
    labels=[labels;categorical(targetStr2)];
end
imdsTrain.Files = files;
imdsTrain.Labels=labels;
labelCount_oversampled = countEachLabel(imdsTrain)

histogram(imdsTrain.Labels)


%% Test SMOTE

labelCount = countEachLabel(imdsTest)
histogram(imdsTest.Labels)
labels=imdsTest.Labels;
[G,classes] = findgroups(labels);
numObservations = splitapply(@numel,labels,G);

desiredNumObservationsPerClass = max(numObservations);

files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},imdsTest.Files,G);
files = vertcat(files{:});
labels=[];info=strfind(files,'\');
for i=1:numel(files)
    idx=info{i};
    dirName=files{i};
    targetStr=dirName(idx(end-1)+1:idx(end)-1);
    targetStr2=cellstr(targetStr);
    labels=[labels;categorical(targetStr2)];
end
imdsTest.Files = files;
imdsTest.Labels=labels;
labelCount_oversampled = countEachLabel(imdsTest)

%% Valid SMOTE
labelCount = countEachLabel(imdsValid)
histogram(imdsValid.Labels)
labels=imdsValid.Labels;
[G,classes] = findgroups(labels);
numObservations = splitapply(@numel,labels,G);

desiredNumObservationsPerClass = max(numObservations);

files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},imdsValid.Files,G);
files = vertcat(files{:});
labels=[];info=strfind(files,'\');
for i=1:numel(files)
    idx=info{i};
    dirName=files{i};
    targetStr=dirName(idx(end-1)+1:idx(end)-1);
    targetStr2=cellstr(targetStr);
    labels=[labels;categorical(targetStr2)];
end
imdsValid.Files = files;
imdsValid.Labels=labels;
labelCount_oversampled = countEachLabel(imdsValid)



%% Load the pre-trained model, VGGNet16

net = vgg16 ; 
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
learnableLayer='fc8';
classLayer='output';
%% Modify the network for the current task
numClasses = numel(categories(imds.Labels));
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,learnableLayer,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer,newClassLayer);


%%%%% OR
% inputSize = net.Layers(1).InputSize
% %Replace Final Layers
% layersTransfer = net.Layers(1:end-3);
% 
% numClasses = numel(categories(imdsTrain.Labels))
% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer
%     classificationLayer];

%% Define image augmenter 
pixelRange = [-30 30];
RotationRange = [-30 30];
scaleRange = [0.8 1.2];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange, ...
    'RandRotation',RotationRange ...
    ); 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
     'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');
augimdsValid = augmentedImageDatastore(inputSize(1:2),imdsValid,"ColorPreprocessing","gray2rgb");
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest,"ColorPreprocessing","gray2rgb");

%% Specify the training options
miniBatchSize = 64;
valFrequency = max(floor(numel(augimdsTest.Files)/miniBatchSize)*10,1);

options = trainingOptions('sgdm', ...
    'Momentum',0.9,...
    'L2Regularization',1e-6,...
    'MiniBatchSize',64, ...
    'MaxEpochs',75, ...
    'InitialLearnRate',0.001, ...
    'LearnRateDropFactor', 0.1,...
    'LearnRateDropPeriod', 5,...
    'LearnRateSchedule','piecewise',...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValid, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',5,...
    'Verbose',true, ...,
    'ExecutionEnvironment','parallel');

[info , net] = trainNetwork(augimdsTrain,lgraph,options);

