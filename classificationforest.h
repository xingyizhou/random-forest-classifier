#ifndef CLASSIFICATIONFOREST_H
#define CLASSIFICATIONFOREST_H

#include"common.h"

class ClassificationForest
{
private:
    TrainParameter *tp;
    TrainData *trainDataPtr;
    std::vector<splitNode *> trees;
    int weakLearnerType;
    int dimensionNum;

    void trainTree(int treeId);
    splitCandidate findBestSplit(Range &range);
    int sortData(Range range, splitCandidate phi);
    bool testPurity(Range range);
    int getRangeLabel(Range range);
    double calculateFeature(Data *data,splitCandidate *phi);

    void writeTree(FILE *fp, int treeId);
    void writeNode(Node *cur,FILE *fp);
    void loadTree(FILE *fp, int weakLearnerType);
    void loadNode(Node **cur,char nodeType,FILE *fp,int demensionNum);
	void deleteTree(Node *root);
public:
    ClassificationForest(){}
    ~ClassificationForest();
    void trainForest(TrainParameter *trainParameter,TrainData *trainData);
    void writeForest(std::string fileName);
    void loadForest(std::string fileName);
    int classification(Data &data,std::vector<int> &votes);

};

#endif // CLASSIFICATIONFOREST_H
