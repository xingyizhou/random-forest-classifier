#ifndef COMMON_H
#define COMMON_H
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<vector>
#include<algorithm>
#include<sstream>
#include<fstream>
#include<stack>
#include<ctime>
#include<cassert>
//#include<omp.h>

#define MAX_DOUBLE (1e300)
#define INF (0x7fffffff)
#define EPS (1e-6)

class TrainParameter
{
public:
    std::string trainDataPath;

    int treeNum;
    int splitFunctionNum;

    int thresholdNum;
    double baggingRate;
    int weakLearnerType;

    int dimentionNum;
    int kmeansIteration;
    int minSample;
    void print()
    {
        std::cout <<"weakLearnerType = "<<weakLearnerType<<std::endl;
        std::cout<<"treeNum = "<<treeNum<<std::endl;
        std::cout<<"splitFunctionNum = "<<splitFunctionNum<<std::endl;
        std::cout<<"thresholdNum = "<<thresholdNum<<std::endl;
        std::cout << "baggingRate = " << baggingRate << std::endl;
    }
};

class splitCandidate
{
public:
    std::vector<int> index;
    std::vector<double> weight;
    double threshold;

    splitCandidate(){}
    ~splitCandidate(){}
};

class Node
{
public:
    Node(){}
    ~Node(){}
    virtual bool isLeaf()=0;
};

class leafNode:public Node
{
public:
    int label;

    leafNode(){}
    ~leafNode(){}

    leafNode(int label):label(label){}

    inline bool isLeaf()
    {
        return true;
    }
};

class splitNode:public Node
{
public:
    Node *left,*right;
    splitCandidate phi;

    splitNode(){}
    ~splitNode(){}

    inline bool isLeaf()
    {
        return false;
    }

};


class Data
{
public:
    std::vector<int> index;
    std::vector<double> value;
    double feature;

    inline double operator[](int u)
    {
        std::vector<int>::iterator itr=std::lower_bound(index.begin(),index.end(),u);
        if (*itr==u) return value[itr-index.begin()];
        return 0;
    }
};


class TrainData
{
public:
    std::vector<Data> data;
    std::vector<int> labels;
    TrainData(){}
    ~TrainData(){}


    void reOrder()
    {
        for (int i=0;i<data.size();i++)
        {
            int j=rand()%(data.size()-i)+i;
            std::swap(data[i],data[j]);
            std::swap(labels[i],labels[j]);
        }
    }

    void getData(std::string s,Data &item,int &label)
    {
        int index=0;
        double value=0;
        bool getLabel=true;
        bool getIdx=false;
        bool getValue=false;
        bool decimalPoint=false;
        double rate=0.1;
        int sig = 1;
        
        if (s[s.size() - 1] != ' ')
          s=s+' ';
        item.value.clear();
        item.index.clear();
        label=0;

        for (int i=0;i<s.size();i++)
        {
            if (s[i]<='9'&&s[i]>='0')
            {
                if (getLabel)
                    label=label*10+(s[i]-'0');
                else if (getIdx)
                    index=index*10+(s[i]-'0');
                else if (decimalPoint)
                {
                    value=value+(s[i]-'0')*rate;
                    rate*=0.1;
                } else
                    value=value*10+(s[i]-'0');
            } else
            if (s[i]==' ')
            {
                if (getLabel) getLabel=false;
                else
                {
                    item.value.push_back(value * sig);
                    sig = 1;
                    value=0;
                }
                getIdx=true;
                getValue=false;
                decimalPoint=false;
                rate=0.1;
            } else
            if (s[i]==':')
            {
                item.index.push_back(index);
                index=0;
                getIdx=false;
                getValue=true;
            } else
            if (s[i]=='.')
                decimalPoint=true;
            else 
            if (getValue && s[i] == '-')
                sig = -1;
        }

    }

    TrainData(std::string trainDataPath,int maxNum=INF)
    {
        std::cout<<"Getting train data from "<<trainDataPath<<"."<<std::endl;
        std::ifstream fin(trainDataPath.c_str());
        Data item;
        int label;
        std::string s;
        while (getline(fin,s))
        {
            getData(s,item,label);
            data.push_back(item);
            labels.push_back(label);
            if (data.size()>=maxNum) break;
        }
		    fin.close();
        //reOrder();
        std::cout<<"Get train data from "<<trainDataPath<<" ok."<<std::endl;
    }
    TrainData(std::vector<Data> data, std::vector<int> labels) :data(data), labels(labels){}

};

class Range
{
public:
    int l, r;
    TrainData *td;
    Range(){}
    ~Range(){}

    Range(TrainData *td, int l, int r) :td(td), l(l), r(r){}
    inline int size()
    {
        return r - l + 1;
    }
};

class StackElement
{
public:
    splitNode *cur;
    Range range;
    int dep;
    StackElement(){}
    StackElement(splitNode *cur, Range range, int dep) :cur(cur), range(range), dep(dep){}
};



inline double randDouble()
{
    return 1.0*rand()/RAND_MAX;
}

inline double sqr(double u)
{
    return u*u;
}



#endif // COMMON_H
