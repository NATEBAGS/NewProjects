#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "pagedir.h"

//Creating the needed .crawler directory
bool pagedir_init(const char *pageDirectory) {
    //Creates a file to mark the directory produced by the crawler, (.crawler)
    char filename[100];
    //Constructs the filename using snprintf
    snprintf(filename, sizeof(filename), "%s/.crawler", pageDirectory);
    //Opens the file to write and prints an error if it doesn't work
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error: Failed to create .crawler file in directory %s\n", pageDirectory);
        return false;
    }
    fclose(fp);
    //Returns true if it was successfully opened
    return true;
}

//Function to save and write the URL,Depth,HTML into the pages 
void pagedir_save(const webpage_t *page, const char *pageDirectory, const int documentID) {
    //Makes sure the inputs are correct
    if (page == NULL || pageDirectory == NULL || documentID < 0) {
        printf("Error: Invalid inputs\n");
        return;
    }
    //Creating the filename
    char filename[100];
    //Initializes the creation of the files that are going to be written to, with its own unique DocId as well
    snprintf(filename, sizeof(filename), "%s/%d", pageDirectory, documentID);
    //Opens the file for writing and prints an error if not
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error: Failed to create file %s\n", filename);
        return;
    }

    //Writing the URL and depth information to the file
    fprintf(fp, "%s\n", webpageURL(page)); //URL here
    fprintf(fp, "%d\n", webpageDepth(page)); //Depth here

    //Writing the HTML to the file
    fprintf(fp, "%s", webpageHTML(page)); //HTML here
    //Makes sure the file is closed properly
    fclose(fp);
}
