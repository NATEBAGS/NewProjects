#ifndef HASH_H_
#define HASH_H_
#include <stdbool.h>
#include <stdio.h>
#include "set.h"

//Hashtable structure implementation
typedef struct hashtable {
    int num_slots;
    set_t **set;
} hashtable_t;

//Hashtable function declarations
hashtable_t *hashtable_new(const int num_slots);
unsigned int string_to_hash(const char *key);
bool hashtable_insert(hashtable_t *ht, const char *key, void *item);
void *hashtable_find(hashtable_t *ht, const char *key);
void hashtable_delete(hashtable_t *ht, void (*itemdelete)(void *item));
void hashtable_print(hashtable_t *ht, FILE *fp,
                     void (*itemprint)(FILE *fp, const char *key, void *item));
void hashtable_iterate(hashtable_t *ht, void *arg,
                       void (*itemfunc)(void *arg, const char *key, void *item));
#endif //HASH_H_
                       
