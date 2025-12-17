#include "set.h"
#include "hashtable.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>


hashtable_t *hashtable_new(const int num_slots) {
    //Allocating memory for the new hashtable
    hashtable_t *new_hash = malloc(sizeof(hashtable_t));
    //Setting up and allocating memory
    new_hash->num_slots = num_slots;
    new_hash->set = malloc(num_slots * sizeof(set_t*));

    //Initialize each set as a new hastable
    for (int i = 0; i < num_slots; i++) {
        new_hash->set[i] = set_new();
    }

    return new_hash;
}
//Added an extra function to calculate the hash value of a key
unsigned int string_to_hash(const char *key) {
    unsigned int hash = 0;
    while (*key != '\0') {
        hash = (hash * 31) + *key;
        key++;
    }
    return hash;
}

bool hashtable_insert(hashtable_t *ht, const char *key, void *item) {
    unsigned int hash = string_to_hash(key) % ht->num_slots;
    return set_insert(ht->set[hash], key, item);
}


void *hashtable_find(hashtable_t *ht, const char *key) {
    //Checking for NULL parameters or empty hashtable
    if (ht == NULL || key == NULL) {
        return NULL;
    }

    //Calculating the hash value
    unsigned int hash = string_to_hash(key) % ht->num_slots;

    //Uses set_find function to find the item key in the set
    return set_find(ht->set[hash], key);
}

void hashtable_print(hashtable_t *ht, FILE *fp, void (*itemprint)(FILE *fp, const char *key, void *item)) {
    //Checking for NULL file pointer or hashtable
    if (fp == NULL) {
        //Nothing is done if NULL
        return;
    }

    if (ht == NULL) {
        //prints (null) as instructed
        fprintf(fp, "(null)\n");
        return;
    }

    //Iterates through the hashtable and print each set 
    for (int i = 0; i < ht->num_slots; i++) {
        //Utilizing set_print for each set in hashtable
        set_print(ht->set[i], fp, itemprint);
    }
}

void hashtable_iterate(hashtable_t *ht, void *arg, void (*itemfunc)(void *arg, const char *key, void *item)) {
    //Checks for NULL hashtable or itemfunc
    if (ht == NULL || itemfunc == NULL) {
        return; //Do nothing if NULL hashtable or itemfunc
    }

    //Iterates through the hashtable and uses itemfunc on each item
    for (int i = 0; i < ht->num_slots; i++) {
        set_iterate(ht->set[i], arg, itemfunc); //Ustilizing previous set_iterate
    }
}

//Function to delete a hashtable
void hashtable_delete(hashtable_t *ht, void (*itemdelete)(void *item)) {
    //Makes sure hashtable is not NULL before deleting
    if (ht != NULL) {
	//Iterating through the hashtable to delete, and if successful the memory is deallocated (freed)
        for (int i = 0; i < ht->num_slots; i++) {
            if (ht->set[i] != NULL) {
                set_delete(ht->set[i], itemdelete);
                ht->set[i] = NULL;
            }
        }
        free(ht->set);  
        free(ht);       
    }
}


