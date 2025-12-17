#ifndef AVL_TREE_H
#define AVL_TREE_H
#include <string>

struct Node {
  std::string key;
  int height;
  Node *left;
  Node *right;

  Node(std::string k) : key(k), height(1),
  left(NULL), right(NULL) {}
};

class AVLTree {
  public:
    AVLTree();
    ~AVLTree();

    // void insert(const std::string &key);
    int rangeQuery(const std::string &low, const std::string &high);
    void clear();
    Node* getRoot() const;
    void insert(const std::string& key);
    void inorderTraversal(Node *root);
    Node* insert(Node* node, const std::string &key);

  private:
  Node* root;
  int getHeight(Node* n);
  Node* rotateLeft(Node* y);
  Node* rotateRight(Node* x);
  int getBalance(Node* N);
  int countRange(Node* node, const std::string &low, const std::string &high);
  void clear(Node* node);
  Node* createNode(const std::string& value);
};

#endif

