#include"classificationforest.h"

using namespace std;

void setTrainParameter(TrainParameter &tp,string trainPatameterPath="trainParameter.txt")
{
    tp.trainDataPath="train.txt";

    FILE *fp=fopen(trainPatameterPath.c_str(),"r");
    if (!fp) return;
    fscanf(fp,"%d",&tp.trainDataNum);
    fscanf(fp,"%d",&tp.weakLearnerType);
    fscanf(fp,"%d",&tp.treeNum);
    fscanf(fp,"%d",&tp.splitFunctionNum);
    fscanf(fp,"%d",&tp.thresholdNum);
    fscanf(fp,"%lf",&tp.baggingRate);
    fclose(fp);
}

double test(ClassificationForest *rdf,int dataNum,string outPutFile="output")
{
    TrainData testData("test.txt",dataNum);
    vector<int> votes;

    int accuracy=0;
    for (int i=0;i<testData.data.size();i++)
    {
        int pred=rdf->classification(testData.data[i],votes);
        if (pred==testData.labels[i]) accuracy++;
    }
    cout<<"Accuracy = "<<1.0*accuracy/testData.data.size()*100<<"% ("<<accuracy<<"/"<<testData.data.size()<<")"<<endl;
	  return 1.0*accuracy/testData.data.size();
}


void recordResult(TrainParameter tp,double trainTime,double Accuracy,string recordPath="record.txt")
{
	string s;
	vector<string> history;
	ifstream fin(recordPath.c_str());
	while (getline(fin,s))
		history.push_back(s);
	fin.close();

	FILE *fp=fopen(recordPath.c_str(),"w");
	for (int i=0;i<history.size();i++)
		fprintf(fp,"%s\n",history[i].c_str());
	fprintf(fp,"%-15d ",DATADIMENTION);
	fprintf(fp,"%-15d ",tp.trainDataNum);
	fprintf(fp,"%-15d ",tp.weakLearnerType);
	fprintf(fp,"%-7d ",tp.treeNum);
	fprintf(fp,"%-19d ",tp.splitFunctionNum);
	fprintf(fp,"%-15d ",tp.thresholdNum);
	fprintf(fp,"%-11f ",tp.baggingRate);
	fprintf(fp,"%-11f ",Accuracy);
	fprintf(fp,"%-.1f min",trainTime);
	fprintf(fp,"\n");
	fclose(fp);
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    ClassificationForest rdf;
    TrainParameter tp;
    time_t t_start, t_end;

    setTrainParameter(tp);

    TrainData trainData = TrainData(tp.trainDataPath,tp.trainDataNum);

    t_start = time(NULL);
    rdf.trainForest(&tp,&trainData);
    t_end = time(NULL);
	  double trainTime=difftime(t_end,t_start)/60.0;
    //rdf.loadForest();

    printf("trainTime = %.0f min\n",trainTime);

    t_start = time(NULL);
    double Accuracy=test(&rdf,tp.trainDataNum);
    t_end = time(NULL);
    printf("testTime = %.0f s\n", difftime(t_end,t_start));
    tp.print();
    recordResult(tp,trainTime, Accuracy);

    return 0;
}
