function [testNum,MPJPE_no_align,MPJPE] = testH36M(seqName,folderName,netName,iterations)

load(['../data_test/GT_3D/',seqName,'.mat']);
load(['../data_test/SH_2D/',seqName,'.mat']);

nJoints = 17;

%translate the position of joints relative to root
GT3D_all = test3D(:,[1 2 3 4 7 8 9 13 14 15 16 18 19 20 26 27 28],:);
GT3D_all = GT3D_all - repmat(GT3D_all(:,1,:),[1 nJoints 1]);
GT3D_all = GT3D_all(:,2:end,:);

testNum = size(GT3D_all,3);

load('../data_test/mean2D.mat');
load('../data_test/mean3D.mat');
%mean sub
for i=1:16
    m = mean2D(:,i);
    testSH2D(:,i,:) = (testSH2D(:,i,:)-repmat(m,[1 1 testNum]));
end

disp(seqName);


%make dep file
modelName = [folderName,'/',netName,'.prototxt'];
phase = 'test';
weights = [folderName,'/net_iter_',num2str(iterations),'.caffemodel'];
net = caffe.Net(modelName, weights, phase);
caffe.set_mode_gpu();

displayIter = 1000;
batchSize = 128;

batchGT3D = zeros(3,nJoints-1,batchSize,'single');
batchGT2D = zeros(2,nJoints-1,batchSize,'single');
loss3DAcc = 0;
curIndex = 1;

iterNum = ceil(testNum/batchSize);
MPJPE = 0;
MPJPE_no_align = 0;
GT = single(zeros(3,16,batchSize));
lastBatchNum = batchSize-(iterNum*batchSize-testNum);
for iter=1:iterNum
    lastBatch = false;
    
    for batchNo=1:batchSize
        if curIndex>testNum
            curIndex = 1;
            lastBatch = true;
        end
        
        batchGT2D(:,:,batchNo) = testSH2D(:,:,curIndex);
        GT(:,:,batchNo) = GT3D_all(:,:,curIndex);
        curIndex = curIndex + 1;
    end
    
    score =  net.forward([{reshape(batchGT2D,32,batchSize)};{reshape(batchGT3D,48,batchSize)}]);
    loss3DAcc = loss3DAcc+score{1};

    result3D = net.blobs('fc3').get_data();
   
    
    for i=1:batchSize
        if lastBatch == true && i>lastBatchNum
            break;
        end
        
        %unnorm
        tempX = [[0;0;0],reshape(result3D(:,i),3,16)+mean3D];
        pts3D = [[0;0;0],GT(:,:,i)];
        %tempX = reshape(result3D(:,i),3,16)+mean3D;
        %pts3D = GT(:,:,i);
                
        pts3D = pts3D';
        tempX = tempX';
        for ii=1:size(pts3D,1)
            MPJPE_no_align = MPJPE_no_align + norm(pts3D(ii,:)-tempX(ii,:));
        end
        [d,normalized3D,transform] = procrustes(pts3D,tempX,'reflection',false,'scaling',true);
        for ii=1:size(pts3D,1)
            MPJPE = MPJPE + norm(pts3D(ii,:)-normalized3D(ii,:));
        end
        

    end
    
    if mod(iter,displayIter)==0
        disp([num2str(iter), ' : ',num2str(MPJPE/(17*batchSize*iter)), ', ',num2str(MPJPE_no_align/(17*batchSize*iter))]);
    end
end

disp(['MPJPE w/ alignment : ',num2str(MPJPE/(17*testNum))]);
disp(['MPJPE w/o alignment : ',num2str(MPJPE_no_align/(17*testNum))]);

caffe.reset_all();