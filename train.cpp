#include "classificationforest.h"

using namespace std;
int LABELNUM = 0;
int DATADIMENTION = 0;


void setTrainParameter(TrainParameter &tp,char *inputFileName, string trainPatameterPath = "trainParameter.txt")
{
    tp.trainDataPath = string(inputFileName);

    FILE *fp = fopen(trainPatameterPath.c_str(),"r");
    if (!fp) {
      tp.weakLearnerType = 1;
      tp.treeNum = 100;
      tp.splitFunctionNum = sqrt(DATADIMENTION + 1);
      tp.thresholdNum = 10;
      tp.baggingRate = 0.6;
      return;
    }
    cout << "Loading train parameters." <<endl;
    fscanf(fp,"%d",&tp.weakLearnerType);
    fscanf(fp,"%d",&tp.treeNum);
    fscanf(fp,"%d",&tp.splitFunctionNum);
    fscanf(fp,"%d",&tp.thresholdNum);
    fscanf(fp,"%lf",&tp.baggingRate);
    fclose(fp);
}

int main(int argc, char **argv) {
  char inputFileName[1024];
  char modelFileName[1024];
  strcpy(inputFileName, argv[1]);
  strcpy(modelFileName, argv[2]);
  
  ifstream fin(inputFileName);
  if (!fin) {
  	cerr << "File not exists!" << endl;
  	return 0;
  }
  
  string s;
  while (getline(fin, s)) {
  	int index = 0;
  	bool getLabel = true;
  	bool getIdx = false;
    int label = 0;
  	s = s + ' ';
    for (int i = 0; i < s.size(); i++) {
      if (s[i] <= '9' && s[i] >= '0') {
      	if (getLabel) 
      		label = label * 10 + (s[i] - '0');
      	else if (getIdx)
      		index = index * 10 + (s[i] - '0');
      } else 
      if (s[i] == ' ') {
        getLabel = false;
        LABELNUM = label > LABELNUM ? label : LABELNUM;
        getIdx = true;
      } else 
      if (s[i] == ':') {
        DATADIMENTION = DATADIMENTION > index ? DATADIMENTION : index;
        index = 0;
        getIdx = false;
      }
    }
  }
  fin.close();
  LABELNUM++;
  DATADIMENTION;


  srand(time(NULL));

  ClassificationForest rdf;
  TrainParameter tp;
  time_t t_start, t_end;

  setTrainParameter(tp, inputFileName);

  TrainData trainData = TrainData(tp.trainDataPath);

  t_start = time(NULL);
  rdf.trainForest(&tp,&trainData);
  rdf.writeForest(string(modelFileName));
  t_end = time(NULL);
  double trainTime=difftime(t_end,t_start)/60.0;
  printf("trainTime = %.0f min\n",trainTime);

}