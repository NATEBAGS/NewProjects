#ifndef WEBPAGE_H
#define WEBPAGE_H

#include <stddef.h>

//Initializing the webpge structure
typedef struct webpage {
    char *url;
    char *html;
    size_t length;
    int depth;
    struct webpage *page;
    struct webpage *next;
    size_t html_len;
} webpage_t;

//Function Declarations here
int webpageDepth(const webpage_t *page);
char *webpageHTML(const webpage_t *page);
char *webpageURL(const webpage_t *page);
void webpageDelete(void *data);
webpage_t *webpageNew(char *url, const int depth, char *html);

#endif /* WEBPAGE_H */

