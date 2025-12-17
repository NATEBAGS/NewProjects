#ifndef PAGEDIR_H
#define PAGEDIR_H
#include "webpage.h"
#include "crawler.h"
//Function to initialize .crawler directory
bool pagedir_init(const char *pageDirectory);

//Funtion to declare function to save the pages on the directory
void pagedir_save(const webpage_t *page, const char *pageDirectory, const int documentID);

#endif /* PAGEDIR_H */
