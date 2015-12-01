#include "classificationforest.h"
using namespace std;


double ClassificationForest::calculateFeature(Data *data, splitCandidate *phi)
{
    double res=0;
    if (weakLearnerType<3)
        for (int i=0;i<phi->index.size();i++)
            res+=phi->weight[i]*(*data)[phi->index[i]];
    else
    {
        res += phi->weight[0] * (*data)[phi->index[0]] * (*data)[phi->index[0]];
        res += phi->weight[1] * (*data)[phi->index[0]] * (*data)[phi->index[1]];
        res += phi->weight[2] * (*data)[phi->index[1]] * (*data)[phi->index[1]];
        res += phi->weight[3] * (*data)[phi->index[0]];
        res += phi->weight[4] * (*data)[phi->index[1]];
    }
    return res;
}

splitCandidate ClassificationForest::findBestSplit(Range &range)
{
    double bestGain=-MAX_DOUBLE;
    splitCandidate bestSplit;

    for (int sf=0;sf<tp->splitFunctionNum;sf++)
    {
        splitCandidate phi;

        vector<double> Gain(tp->thresholdNum);
        vector<double> threshold(tp->thresholdNum);

        if (tp->weakLearnerType<3)
        {
            for (int d=0;d<tp->weakLearnerType;d++)
            {
                phi.index.push_back(rand()%DATADIMENTION+1);
                phi.weight.push_back(randDouble()*2-1);
            }
        } else
        if (tp->weakLearnerType==3)
        {
            for (int d=0;d<2;d++)
                phi.index.push_back(rand()%DATADIMENTION+1);
            for (int i=0;i<5;i++)
                phi.weight.push_back(randDouble()*2-1);
        }


        double maxFeature=-MAX_DOUBLE;
        double minFeature=MAX_DOUBLE;
        for (int i=range.l;i<=range.r;i++)
        {
            (range.td->data[i]).feature=calculateFeature(&(range.td->data[i]),&phi);
            maxFeature=max(maxFeature,(range.td->data[i]).feature);
            minFeature=min(minFeature,(range.td->data[i]).feature);
        }

        if (fabs(maxFeature-minFeature)<EPS)
        {
            sf--;
            continue;
        }


        for (int t=0;t<tp->thresholdNum;t++)
        {
            int cardL=0,cardR=0;
            double HL=0,HR=0;
            vector<int> labelProbL(LABELNUM);
            vector<int> labelProbR(LABELNUM);


            threshold[t]=randDouble()*(maxFeature-minFeature)+minFeature;//

            for (int i=range.l;i<=range.r;i++)
            {
                if ((range.td->data[i]).feature<=threshold[t])
                {
                    cardL++;
                    labelProbL[range.td->labels[i]]++;

                } else
                {
                    cardR++;
                    labelProbR[range.td->labels[i]]++;
                }
            }

            for (int i=0;i<LABELNUM;i++)
            {
                if (labelProbL[i]!=0&&labelProbL[i]!=cardL)
                    HL+=1.0*labelProbL[i]/cardL*log2(1.0*labelProbL[i]/cardL);
                if (labelProbR[i]!=0&&labelProbR[i]!=cardR)
                    HR+=1.0*labelProbR[i]/cardR*log2(1.0*labelProbR[i]/cardR);            
            }
            Gain[t]=cardL*HL+cardR*HR;
        }

        for (int t=0;t<tp->thresholdNum;t++)
        {
            if (Gain[t]>bestGain)
            {
                bestGain=Gain[t];
                phi.threshold=threshold[t];
                bestSplit=phi;
            }
        }

    }
    //cerr<<"bestGain = "<<bestGain<<endl;
    return bestSplit;
}


int ClassificationForest::sortData(Range range, splitCandidate phi)
{
    int idx;
    bool findRight=false;

    for (int i=range.l;i<=range.r;i++)
    {
        bool goLeft;
        
        goLeft=calculateFeature(&(range.td->data[i]),&phi)<=phi.threshold;

        if (!findRight&&!goLeft)
        {
            findRight=true;
            idx=i;
        } else
        if (findRight&&goLeft)
        {
            swap(range.td->data[i],range.td->data[idx]);
            swap(range.td->labels[i],range.td->labels[idx]);
            idx++;
        }
    }

    return idx;
}


bool ClassificationForest::testPurity(Range range)
{
    int i;
    int firstLabel = range.td->labels[range.l];
    for (i=range.l+1;i<=range.r;i++)
        if (range.td->labels[i]!=firstLabel) break;
    if (i>range.r)
        return true;
    return false;
}

int ClassificationForest::getRangeLabel(Range range)
{
    return range.td->labels[range.l];
}


void ClassificationForest::trainTree(int treeId)
{
    splitNode **root,*cur;
    Range range,tmpRange;
    stack<StackElement> stk;
    int dep,mid;

    cout<<"Training tree "<<treeId<<"."<<endl;
    TrainData *td=new TrainData(trainDataPtr->data,trainDataPtr->labels);
    td->reOrder();
    range.td = td;
    range.l=0;
    range.r=(td->data.size()-1)*tp->baggingRate;

    root=&trees[treeId];
    *root=new splitNode();

    stk.push(StackElement(*root,range,0));

    while (!stk.empty())
    {
        cur=stk.top().cur;
        range=stk.top().range;
        dep=stk.top().dep;

        stk.pop();
        //cout<<"Running range "<<range.l<<" "<<range.r<<"."<<endl;

        cur->phi=findBestSplit(range);
        //cerr<<"findOK"<<endl;
        mid=sortData(range,cur->phi);
        //cerr<<"sortOK"<<endl;
        tmpRange=Range(range.td,range.l,mid-1);
        if (testPurity(tmpRange))
            cur->left=new leafNode(getRangeLabel(tmpRange));
        else
        {
            cur->left=new splitNode();
            stk.push(StackElement((splitNode *)cur->left,tmpRange,dep+1));
        }

        tmpRange=Range(range.td,mid,range.r);
        if (testPurity(tmpRange))
            cur->right=new leafNode(getRangeLabel(tmpRange));
        else
        {
            cur->right=new splitNode();
            stk.push(StackElement((splitNode *)cur->right,tmpRange,dep+1));
        }
    }
    delete td;
}

void ClassificationForest::trainForest(TrainParameter *trainParameter,TrainData *trainData)
{
    cout<<"Start training forest."<<endl;
    tp=trainParameter;
    trainDataPtr=trainData;

    weakLearnerType=tp->weakLearnerType;
    trees.resize(tp->treeNum,NULL);

#pragma omp parallel for
    for(int i=0;i<tp->treeNum;i++)
        trainTree(i);

    writeForest();
    cout<<"Train forest ok."<<endl;
}

int ClassificationForest::classification(Data &data,vector<int> &votes)
{
    votes=vector<int>(LABELNUM,0);

    for (int t=0;t<trees.size();t++)
    {
        Node *cur=(Node *)trees[t];
        splitNode *sNode;
        while (!(cur->isLeaf()))
        {
            sNode=(splitNode *)cur;
            double feature=calculateFeature(&data,&(sNode->phi));
            cur=feature <= sNode->phi.threshold ? sNode->left : sNode->right;
        }
        votes[((leafNode *)cur)->label]++;
    }

    int res=0;
    for (int i=0;i<LABELNUM;i++)
        if (votes[i]>votes[res])
            res=i;
    return res;
}


void ClassificationForest::writeNode(Node *cur,FILE *fp)
{
    if (cur->isLeaf())
    {
        fprintf(fp,"L");
        fprintf(fp," %d\n",((leafNode *)cur)->label);
    } else
    {
        splitCandidate phi=((splitNode *)cur)->phi;
        fprintf(fp,"S");
        for (int i=0;i<phi.index.size();i++)
            fprintf(fp," %d",phi.index[i]);
        for (int i = 0; i < phi.weight.size();i++)
            fprintf(fp, " %f", phi.weight[i]);
        fprintf(fp," %lf",phi.threshold);
		fprintf(fp,"\n");
    }
}

void ClassificationForest::writeTree(int treeId,std::string fileName)
{
    FILE *fp;
    splitNode *cur;
    stack<splitNode *>stk;

    fp=fopen(fileName.c_str(),"w");
    //cout<<"Writing tree to file "<<fileName<<"."<<endl;

    writeNode((Node *)trees[treeId],fp);

    stk.push(trees[treeId]);
    while (!stk.empty())
    {
        cur=stk.top();
        stk.pop();

        writeNode(cur->left,fp);
        if (!(cur->left->isLeaf())) stk.push((splitNode *)cur->left);

        writeNode(cur->right,fp);
        if (!(cur->right->isLeaf())) stk.push((splitNode *)cur->right);
    }
    fclose(fp);
}

void ClassificationForest::writeForest(string fileName)
{
    cout<<"Writing forest."<<endl;
    FILE *fp=fopen((fileName+string("/forestConfigure.txt")).c_str(),"w");
    fprintf(fp,"%d %d\n",tp->treeNum,tp->weakLearnerType);
    fclose(fp);
    for (int i=0;i<trees.size();i++)
    {
        stringstream fileNameSS;
        fileNameSS<<fileName<<"/"<<i<<".tree";
        writeTree(i,fileNameSS.str());
    }
}


void ClassificationForest::loadNode(Node **cur, char nodeType, FILE *fp, int weakLeanerType)
{
    if (nodeType=='L')
    {
        double dist;
        *cur=new leafNode();
        fscanf(fp, "%d", &(((leafNode *)(*cur))->label));
    } else
    {
        int idx;
        double weight;
        *cur=new splitNode();
        
        if (weakLeanerType < 3)
        {
            for (int i = 0; i < weakLeanerType; i++)
            {
                fscanf(fp, "%d ", &idx);
                ((splitNode *)(*cur))->phi.index.push_back(idx);
            }
            for (int i = 0; i < weakLeanerType; i++)
            {
                fscanf(fp, "%lf ", &weight);
                ((splitNode *)(*cur))->phi.weight.push_back(weight);
            }
        }
        else
        {
            fscanf(fp, "%d ", &idx);
            ((splitNode *)(*cur))->phi.index.push_back(idx);
            fscanf(fp, "%d ", &idx);
            ((splitNode *)(*cur))->phi.index.push_back(idx);
            for (int i = 0; i < 5; i++)
            {
                fscanf(fp, "%lf ", &weight);
                ((splitNode *)(*cur))->phi.weight.push_back(weight);
            }
        }
        fscanf(fp, "%lf", &(((splitNode *)(*cur))->phi.threshold));
        
    }
    fscanf(fp,"\n");
}

void ClassificationForest::loadTree(string fileName, int weakLeanerType)
{
    FILE *fp;
    stack<splitNode *> stk;
    splitNode *cur;
    Node *left,*right;
    char nodeType;

    if ((fp=fopen(fileName.c_str(),"r"))==NULL)
    {
        printf("Cannot open file %s.\n",fileName.c_str());
        exit(0);
    } else
    //cout<<"Loading tree from"<<fileName<<"."<<endl;

    fscanf(fp,"%c ",&nodeType);
    loadNode((Node **)&cur, nodeType, fp, weakLeanerType);
    trees.push_back(cur);

    stk.push(cur);
    while (!stk.empty())
    {
        cur=stk.top();
        stk.pop();

        fscanf(fp,"%c ",&nodeType);
        loadNode(&left, nodeType, fp, weakLeanerType);
        cur->left=left;
        if (nodeType=='S')
            stk.push((splitNode *)left);

        fscanf(fp,"%c ",&nodeType);
        loadNode(&right, nodeType, fp, weakLeanerType);
        cur->right=right;
        if (nodeType=='S')
            stk.push((splitNode *)right);
    }
    fclose(fp);
}

void ClassificationForest::loadForest(string fileName)
{
    int treeNum;
    cout<<"Loading forest."<<endl;
    FILE *fp=fopen((fileName+string("/forestConfigure.txt")).c_str(),"r");
    fscanf(fp, "%d %d", &treeNum, &weakLearnerType);
    fclose(fp);

    for (int i=0;i<treeNum;i++)
    {
        stringstream fileNameSS;
        fileNameSS<<fileName<<"/"<<i<<".tree";
        loadTree(fileNameSS.str(), weakLearnerType);
    }
}

void ClassificationForest::deleteTree(Node *root)
{
    if (!root) return;
    if (!(root->isLeaf()))
    {
        deleteTree(((splitNode *)root)->left);
        deleteTree(((splitNode *)root)->right);
    }
    delete root;
}

ClassificationForest::~ClassificationForest()
{
    for (int i=0;i<trees.size();i++)
        deleteTree((Node *)trees[i]);
}





