#ifndef SET_H_
#define SET_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Defining the node struct
typedef struct node {
    char *key;
    void *item;
    struct node *next;
} node_t;

// Defining the set 
typedef struct set {
    node_t *head;
} set_t;

set_t* set_new(void);
bool set_insert(set_t *set, const char *key, void *item);

void *set_find(set_t *set, const char *key);

void set_print(set_t *set, FILE *fp,
               void (*itemprint)(FILE *fp, const char *key, void *item) );

void set_iterate(set_t *set, void *arg,
                 void (*itemfunc)(void *arg, const char *key, void *item) );

void set_delete(set_t *set, void (*itemdelete)(void *item) );

#endif //SET_H_

