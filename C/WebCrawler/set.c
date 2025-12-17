#include <string.h>
#include "set.h"
#include <stdio.h>
#include <stdlib.h>


set_t* set_new(void) {
    //Allocating memory for set_t
    set_t *set = (set_t *)malloc(sizeof(set_t));
    //checking if memory allocation was successful, returns null if not
    if (set == NULL) {
        return NULL;
    }
    //Initializes the set as empty
    set->head = NULL;
    return set;
}

bool set_insert(set_t *set, const char *key, void *item) {
    // Checks for valid parameters, or returns false
    if (set == NULL || key == NULL || item == NULL) {
        return false;
    }
    node_t *current = set->head;
    //Traverses the set until it reaches the end
    while (current != NULL) {
        //If a duplicate key is found it returns false
        if (strcmp(current->key, key) == 0) {
            return false;
        }
        //Moving the current pointer to the next node
        current = current->next;
    }

    // if no duplicates were found it allocates memory for the new node
    node_t *new_node = (node_t *)malloc(sizeof(node_t));
    if (new_node == NULL) {
        // Returns false if memory allocation fails
        return false;
    }

    //This key is copying the key and assigning it to the item
    new_node->key = strdup(key);
    new_node->item = item;
    new_node->next = set->head;
    //Adds it to the beggining of the linked list
    set->head = new_node;
    //Returns true if successfully inserted
    return true;
}


void *set_find(set_t *set, const char *key) {
    // Check for valid parameters/empty set
    if (set == NULL || key == NULL || set->head == NULL) {
        return NULL;
    }

    // Traversing the set
    node_t *current = set->head;
    while (current != NULL) {
        //If keys is found then it is returned
        if (strcmp(current->key, key) == 0) {
            return current->item;
        }
        //Continues traversing
        current = current->next;
    }
    //returns NULL if no matching key is found
    return NULL;
}

void set_print(set_t *set, FILE *fp, void (*itemprint)(FILE *fp, const char *key, void *item)) {
    // If set is NULL,it then it prints null
    if (set == NULL) {
        if (fp != NULL) {
            fprintf(fp, "(null)\n");
        }
        return;
    }

    // If fp is NULL, nothing happens
    if (fp == NULL) {
        return;
    }

    // If itemprint is NULL or the set is empty, it prints set with no items
    if (itemprint == NULL) {
        fprintf(fp, "(null)\n");
        return;
    }

    // Traverse the set and prints each (key, item) using itemprint function, each on a newline
    node_t *current = set->head;
    while (current != NULL) {
        itemprint(fp, current->key, current->item);
        current = current->next;
    }
}
void set_iterate(set_t *set, void *arg, void (*itemfunc)(void *arg, const char *key, void *item)) {
    // If the set is NULL or itemfunc is NULL, nothing happens
    if (set == NULL || itemfunc == NULL) {
        return;
    }

    // Traverse the set until null and calls the itemfunc on each item
    node_t *current = set->head;
    while (current != NULL) {
        itemfunc(arg, current->key, current->item);
        current = current->next;
   }
}
void set_delete(set_t *set, void (*itemdelete)(void *item)) {
	if (set != NULL) {
		for (node_t *node = set->head; node != NULL; ) {
			if (itemdelete != NULL) { //Checks for Null item before deletion.
				(*itemdelete)(node->item); // delete node's item
			}
			node_t *next = node->next;
			free(node->key);//Freeing the key then the node holding the key
			free(node); 
			node = next; //Iterating through using next
		}

		free(set); //Freeing up the allocated set memory
	}
}
