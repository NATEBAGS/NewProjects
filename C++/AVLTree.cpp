#include "AVLTree.h"
#include <iostream>

using namespace std;
//Constructor and destructor to manage memory
AVLTree::AVLTree() : root(NULL) {}
AVLTree::~AVLTree() { 
  clear(); 
}


void AVLTree::clear(Node* node) {
  if (node) {
    clear(node->left);
    clear(node->right);
    delete node;
  }
}
int AVLTree::rangeQuery(const std::string &low, const std::string &high) {
  return countRange(root, low, high);
}

// Function gets height of the tree
int AVLTree::getHeight(Node *node) {
  if (node == nullptr) {
    return 0;
  }
  return node->height;
}

int getMax(int a, int b) {
  // Gets the maximum of two integers
  return (a > b) ? a : b;
}

Node* AVLTree::createNode(const string& value) {
  return new Node(value);
}

Node* AVLTree::insert(Node* node, const std::string &key) {
   if (node == nullptr) {
    if (root == nullptr) {
      root = createNode(key);
      return root;
    }
    else {
      return createNode(key);
    }
   }
   if (key < node->key) {
     node->left = insert(node->left, key);
   }
   else if (key > node->key) {
     node->right = insert(node->right, key);
   }
   else {
     return node;
   }
   node->height = 1 + getMax(getHeight(node->left), getHeight(node->right));
   // Balance the tree

   int balance = getBalance(node);

   // if its unbalanced, then there are 4 cases 

   // Left Left
   if (balance > 1 && key < node->left->key) {
    return rotateRight(node);
   }
   // Right Right
   if (balance < -1 && key > node->right->key) {
     return rotateLeft(node);
   }
   // Left Right
   if (balance > 1 && key < node->left->key) {
    node->left = rotateLeft(node->left);
    return rotateRight(node);
   }
   // Right Left
   if (balance < -1 && key < node->right->key) {
    node->right = rotateRight(node->right);
    return rotateLeft(node);
   }

   return node;
}

Node* AVLTree::getRoot() const {
  return root;
}

// Left rotation
Node* AVLTree::rotateLeft(Node *x) {
  Node *y = x->right;
  Node *temp = y->left;
 
 // Perform the Rotation
  y->left = x;
  x->right = temp;

  // Update our heights
  x->height = getMax(getHeight(x->left), getHeight(x->right)) + 1;
  y->height = getMax(getHeight(y->left), getHeight(y->right)) + 1;
 
  // Return the new root
  return y;
}

// Right rotation
Node* AVLTree::rotateRight(Node *y) {
  Node *x = y->left;
  Node *temp = x->right;
 
 // Perform the Rotation
  x->right = y;
  y->left = temp;

  // Update our heights
  y->height = getMax(getHeight(y->left), getHeight(y->right)) + 1;
  x->height = getMax(getHeight(x->left), getHeight(x->right)) + 1;
 
  // Return the new root
  return x;
}

int AVLTree::getBalance(Node *N) {
  if (N == nullptr) {
    return 0;
  }
  return getHeight(N->left) - getHeight(N->right);
}

void AVLTree::inorderTraversal(Node *root) {
  if (root != nullptr) {
    inorderTraversal(root->left);
    cout << root->key << " ";
    inorderTraversal(root->right);
  }
}

int AVLTree::countRange(Node* node, const std::string &low, const std::string &high) {
  if (!node) {
    return 0;
  }

  if (node->key >= low && node->key <= high) {
    return 1 + countRange(node->left, low, high) + countRange(node->right, low, high);
  }
  else if (node->key < low) {
    return countRange(node->right, low, high);
  }
  else {
    return countRange(node->left, low, high);
  }
}
void AVLTree::clear() {
  clear(root);
  root = NULL;
}

void AVLTree::insert(const string& key) {
  root = insert(root, key);
}

