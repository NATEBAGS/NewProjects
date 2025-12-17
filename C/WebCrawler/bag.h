#ifndef BAG_H
#define BAG_H
#include "webpage.h"
//Implementation of bag structures
typedef struct bag_node {
    webpage_t *page;
    struct bag_node *next;
} bag_node_t;

typedef struct {
    bag_node_t *head;
    size_t size;
} bag_t;
//Declaring bag functions used
size_t bagSize(bag_t *bag);
void bagDelete(bag_t *bag);
bag_t *bagNew();
void bagInsert(bag_t *bag, webpage_t *page);
webpage_t *bagGrab(bag_t *bag);
bool bagEmpty(bag_t *bag);

#endif /* BAG_H */
