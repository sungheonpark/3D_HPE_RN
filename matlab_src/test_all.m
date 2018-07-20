function [MPJPE_all, MPJPE_no_align_all] = test_all(folderName,netName,iter)
seqNames = {'Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting','SittingDown',...
    'Smoking','Waiting','WalkDog','Walking','WalkTogether'};
testNum = 0;
MPJPE_all = 0;
MPJPE_no_align_all = 0;
for i=1:size(seqNames,2)
    [num, MPJPE_no_align, MPJPE] = testH36M(seqNames{i},folderName,netName,iter);
    testNum = testNum + num;
    MPJPE_all = MPJPE_all + MPJPE;
    MPJPE_no_align_all = MPJPE_no_align_all + MPJPE_no_align;
end
MPJPE_all = MPJPE_all/(17*testNum);
MPJPE_no_align_all = MPJPE_no_align_all/(17*testNum);
disp('Avg');
disp(['MPJPE w/ alignment : ',num2str(MPJPE_all)]);
disp(['MPJPE w/o alignment : ',num2str(MPJPE_no_align_all)]);