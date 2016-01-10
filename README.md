# random-forest-classifier

The same interface as liblinear

##install

make

##demo

./train heart_scale_train heart_scale_train.model
./predict heart_scale_test heart_scale_train.model heart_scale_test.output

##try other train parameters in trainParameter.txt
weakLearnerType // 1, 2, 3
treeNum
splitFunctionNum
thresholdNum
baggingRate
