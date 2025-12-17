#include <fstream>
#include <iostream>
#include <sstream>
#include "AVLTree.h"

using namespace std;

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "Usage " << argv[0] << " <input file> <output file" << endl;
    return 1; // Unsuccessful
  }

  ifstream inputFile(argv[1]);
  ofstream outputFile(argv[2]);
  AVLTree tree;

  string line, task, word1, word2;
  while (getline(inputFile, line)) {
    istringstream iss(line);
    iss >> task;
    if (task == "i") {
      iss >> word1;
      // cout << "The word is:" << word1 << endl;
      tree.insert(word1);
    }
    else if (task == "r") {
      iss >> word1 >> word2;
      // cout << "The words are:" << word1 << " " << word2 << endl;
      int count = tree.rangeQuery(word1, word2);
      outputFile << count << endl;
    }
  }
  tree.clear();

  return 0; // Program executed cleanly
} // Main Scope