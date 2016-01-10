#include "classificationforest.h"

using namespace std;
int LABELNUM = 0;
int DATADIMENTION = 0;

void Predict(ClassificationForest *rdf, char* testFile, char *outputFile) {
  TrainData testData = TrainData(string(testFile));
  vector<int> votes;

  int tot = testData.data.size();
  int accuracy = 0;
  ofstream fout(outputFile);
  for (int i = 0; i < tot; i++) {
  	int pred = rdf->classification(testData.data[i], votes);
  	fout << pred << endl;
  	if (pred == testData.labels[i]) 
      accuracy++;
  }
  fout.close();
  printf("Accuracy = %.2f%% (%d/%d) \n", 1.0 * accuracy / tot, accuracy, tot);
}


int main(int argc, char **argv) {
  char testFile[1024];
  char modelFile[1024];
  char outputFile[1024];
  strcpy(testFile, argv[1]);
  strcpy(modelFile, argv[2]);
  strcpy(outputFile, argv[3]);

  ifstream fin(testFile);
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
  DATADIMENTION++;


  srand(time(NULL));

  ClassificationForest rdf;
  time_t t_start, t_end;

  rdf.loadForest(string(modelFile));

  t_start = time(NULL);
  Predict(&rdf, testFile, outputFile);
  t_end = time(NULL);
  printf("test time = %.0f s\n",difftime(t_end, t_start));
  return 0;
}