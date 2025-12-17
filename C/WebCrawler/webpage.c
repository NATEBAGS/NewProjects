#include <string.h>
#include <stdlib.h>
#include "webpage.h"

//Gets the depth of the page or returns 0
int webpageDepth(const webpage_t *page) { return page ? page->depth : 0;}
//Gets HTML of webpage or returns NULL
char *webpageHTML(const webpage_t *page) { return page ? page->html : NULL;}
//Gets URl of webpage or returns NULL
char *webpageURL(const webpage_t *page) { return page ? page->url : NULL;}

void webpageDelete(void *data) {
  webpage_t *page = data;
  //Mkes sure not to free NULL values, then proceeds to free the data stored from the webpage
  if (page != NULL) {
    if (page->url) free(page->url);
    if (page->html) free(page->html);
    free(page);
  }
}
//Makes a new webpage with all the HTML on it
webpage_t *webpageNew(char *url, const int depth, char *html) {
  //Makes sure to check for errors before creating a webpage
  if (url == NULL || depth < 0) {
    return NULL;
  }
  //Allocating memory for the pages
  webpage_t *page = malloc(sizeof(webpage_t));
  if (page == NULL) {
    return NULL;
  }
  //Allocate memory for the URL and NULL value at the end
  page->url = malloc(strlen(url) + 1);
  if (page->url == NULL) {
    free(page);
    return NULL;
  }
  strcpy(page->url, url);

  page->depth = depth;

  if (html != NULL) {
    page->html = malloc(strlen(html) + 1);
    if (page->html == NULL) {
      free(page->url);
      free(page);
      return NULL;
    }
    //copying the html onto page
    strcpy(page->html, html);
    page->html_len = strlen(html);
    //If it fails it just returns an empty page
  } else {
      page->html = NULL;
      page->html_len = 0;
  }
  //Return new webpage at the end
  return page;
}

