#include <stdlib.h>
#include <stdbool.h>
#include "webpage.h"
#include "bag.h"

//Creating a bag to store webpages
bag_t *bagNew() {
    //Allocating the memory for the bag
    bag_t *bag = (bag_t *)malloc(sizeof(bag_t));
    if (bag == NULL) {
        //Handles memory allocation if it fails
        return NULL;
    }
    //Initializing its components
    bag->head = NULL;
    bag->size = 0;
    return bag;
}

//Function to insert an item into the bag
void bagInsert(bag_t *bag, webpage_t *page) {
    //Allocate memory for the node insert
    bag_node_t *new_node = (bag_node_t *)malloc(sizeof(bag_node_t));
    if (new_node == NULL) {
        //Handles memory allocation if it fails
        return;
    }
    //Handles adjustments based on inserted page
    new_node->page = page;
    new_node->next = bag->head;
    bag->head = new_node;
    bag->size++;
}
//Grabs an item from the bag
webpage_t *bagGrab(bag_t *bag) {
    if (bag->head == NULL) {
        //Mkes sure bag is not empty
        return NULL;
    }
    //handles adjustments and returns page
    bag_node_t *temp = bag->head;
    webpage_t *page = temp->page;
    bag->head = temp->next;
    free(temp);
    bag->size--;
    return page;
}

//Checks if the bag is empty
bool bagEmpty(bag_t *bag) {
	return (bag->head == NULL);
}
//Returns size of the bag
size_t bagSize(bag_t *bag) {
    if (bag == NULL) {
        return 0; //Makes sure bag is not of length 0
    }
    return bag->size;
}

void bagDelete(bag_t *bag) {
    if (bag == NULL) {
        return; //Nothing to delete if the bag is NULL
    }
    //Iterating through the bag until the end
    bag_node_t *current = bag->head;
    while (current != NULL) {
        bag_node_t *temp = current;
        current = current->next;

        //Free any memory associated with the bag
        webpageDelete(temp->page);
        free(temp);
    }
}    
