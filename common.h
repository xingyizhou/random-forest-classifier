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
#include<omp.h>

#define MAX_DOUBLE 1e300
#define INF 0x7fffffff
#define EPS 1e-6

#define LABELNUM 200
#define DATADIMENTION (4096 * 3 * 3)


class TrainParameter
{
public:
    std::string trainDataPath;

    int treeNum;
    int trainDataNum;
    int splitFunctionNum;

    int thresholdNum;
    double baggingRate;
    int weakLearnerType;

    int dimentionNum;
    int kmeansIteration;
    int minSample;
    void print()
    {
        std::cout<<"trainDataNum = "<<trainDataNum<<std::endl;
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
                    item.value.push_back(value);
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
            label--;
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


class Point
{
public:
    std::vector<double> p;

    Point(){}
    ~Point(){}
    Point(std::vector<int> *index,Data *data)
    {
        for (int i=0;i<index->size();i++)
            p.push_back((*data)[(*index)[i]]);
    }

    Point(int u)
    {
        p.resize(u,0);
    }

    double dis(Point u)
    {
        double res=0;
        for(int i=0;i<p.size();i++)
            res+=(p[i]-u[i])*(p[i]-u[i]);
        return res;
    }

    Point operator +(Point &u)
    {
        Point res;
        for (int i=0;i<p.size();i++)
            res.p.push_back(p[i]+u[i]);
        return res;
    }

    Point operator /(double u)
    {
        Point res;
        for (int i=0;i<p.size();i++)
            res.p.push_back(p[i]/u);
        return res;
    }

    double operator[](int u)
    {
        return p[u];
    }

    void print()
    {
        std::cerr<<"Point : ";
        for (int i=0;i<p.size();i++)
            std::cerr<<p[i]<<" ";
        std::cerr<<std::endl;
    }

    bool zero()
    {
        for (int i=0;i<p.size();i++)
            if (fabs(p[i])>EPS) return false;
        return true;
    }

    bool equal(Point &u)
    {
        for (int i=0;i<p.size();i++)
            if (fabs(p[i]-u[i])>EPS) return false;
        return true;
    }

};

class Kmeans
{
public:
    Point p[2];
    std::vector<Point> data;

    Kmeans(){}
    ~Kmeans(){}

    Kmeans(Range &range,std::vector<int> index)
    {
        for (int i=range.l;i<=range.r;i++)
            data.push_back(Point(&index,&(range.td->data[i])));

        p[0]=Point(&index,&(range.td->data[range.l]));
        for (int i=range.l+1;i<=range.r;i++)
            if (range.td->labels[i]!=range.td->labels[range.l])
            {
                p[1]=Point(&index,&(range.td->data[i]));
                break;
            }
    }

    void solve(Range &range,int iterations)
    {
        for (int itr=0;itr<iterations;itr++)
        {
            Point m[2];
            for (int i=0;i<2;i++)
                m[i].p.resize(p[0].p.size(),0);
            int card[2]={0,0};

            for (int i=0;i<data.size();i++)
            {
                if (p[0].dis(data[i])<p[1].dis(data[i]))
                {
                    m[0]=m[0]+data[i];
                    card[0]++;
                } else
                {
                    m[1]=m[1]+data[i];
                    card[1]++;
                }
            }

            if (card[0]==0||card[1]==0)
            {
                std::cerr<<"dataSize = "<<data.size()<<std::endl;
                p[0].print();
                p[1].print();

            }

            assert(card[0]>0&&card[1]>0);
            p[0]=m[0]/card[0];
            p[1]=m[1]/card[1];
        }
    }



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
