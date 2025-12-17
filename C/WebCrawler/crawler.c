#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "hashtable.h"
#include "crawler.h"
#include "curl.h"
#include "url.h"
#include "set.h"
#include "bag.h"
#include "webpage.h"
#include <string.h>

//Created this function to delete items from the structures
void deleteItem(void *item) {
    set_delete((set_t *)item, NULL);
}
//Makes sure that there is the correct number of arguments being passed and within ranges
void parseArgs(const int argc, char *argv[], char **seedURL, char **pageDirectory, int *maxDepth) {
    //Checks if there are only 4 arguments
    if (argc != 4) {
	//If incorrect prints the standard error function
        fprintf(stderr, "Try out: ./crawler seedURL pageDirectory maxDepth\n");
        exit(1);
    }
    //Identifies the parameters
    *seedURL = argv[1];
    *pageDirectory = argv[2];
    *maxDepth = atoi(argv[3]);

    //Additional validation for maxDepth range [0, 10]
    if (*maxDepth < 0 || *maxDepth > 10) {
        fprintf(stderr, "Invalid!: maxDepth's range should be within [0, 10]\n");
        exit(1);
    }
}
//Function checking if URL has already appeared in the visitedURLs hashtable (No duplicate files)
bool isVisited(hashtable_t *visitedURLs, const char *url) {
    //Returns true if the URL is found, otherwise it returns false
    return hashtable_find(visitedURLs, url) != NULL;
}

void crawl(char *seedURL, char *pageDirectory, const int maxDepth) {
    //Initializes hashtable, bag, and document counters to get unique ID's
    hashtable_t *pagesSeen = hashtable_new(100);
    bag_t *pagesToCrawl = bagNew();
    int documentCounter = 0;

    //Initializes a hashtable to store already visited URLs
    hashtable_t *visitedURLs = hashtable_new(100);
    //Allocating memory for the pages
    webpage_t *seedPage = (webpage_t *)malloc(sizeof(webpage_t));
    if (seedPage != NULL) {
        seedPage->url = strdup(seedURL); //copies the seedURL
        //Check if strdup failed to allocate memory and if it does it exits
        if (seedPage->url == NULL) {
            free(seedPage);
            exit(1);
        }
	//Sets the default options to the page
        seedPage->html = NULL;
        seedPage->length = 0;
        seedPage->depth = 0;
    } else {
        exit(1);
    }

    //Inserting the seedPage into the bag
    bagInsert(pagesToCrawl, seedPage);

    //Crawler loop doesn't end until the bag for pages to be crawled is empty
    while (!bagEmpty(pagesToCrawl)) {
	//Grabs a page for crawling
        webpage_t *currentPage = bagGrab(pagesToCrawl);

        //Checks if the current URL has been visited before
        if (!isVisited(visitedURLs, webpageURL(currentPage))) {
            //If URL has not been visited, it proceeds with downloading the content
            size_t htmlSize;
            char *htmlContent = download(webpageURL(currentPage), &htmlSize);
            if (htmlContent == NULL) {
		//If its NUL then it prints with the standard error output 
                fprintf(stderr, "Error in getting the URL: %s\n", webpageURL(currentPage));
                exit(1);
            }

            //Assigns the downloaded content to the webpage's html field
            currentPage->html = htmlContent;
            currentPage->length = htmlSize;

            //Save the webpage to the pageDirectory using the pagedir_save function
            pagedir_save(currentPage, pageDirectory, documentCounter++);

            //If depth is less than the max, it scans for links and adds it to pagesToCrawl
            if (webpageDepth(currentPage) < maxDepth) {
		printf("%d  Fetched: %s\n", webpageDepth(currentPage), webpageURL(currentPage));
                printf("%d  Scanning: %s\n", webpageDepth(currentPage), webpageURL(currentPage));
                pageScan(currentPage, pagesToCrawl, pagesSeen);
            }
	    //If its not already found then it adds it to the visited hashtable
	    if (!hashtable_find(visitedURLs, webpageURL(currentPage))) {
                printf("%d     Found: %s\n", webpageDepth(currentPage), webpageURL(currentPage));
                printf("%d     Added: %s\n", webpageDepth(currentPage), webpageURL(currentPage));
		hashtable_insert(visitedURLs, webpageURL(currentPage), NULL);
            }
        } else {
            //If URL has been visited, skip the downloading
            printf("Skipping already visited URL: %s\n", webpageURL(currentPage));
        }

        //Cleaning up the current page
        if (currentPage->html != NULL) {
            free(currentPage->html);
            currentPage->html = NULL;
        }
	//Deletes the whole page
        webpageDelete(currentPage);
    }

    //Cleanups up everything once its all said and done
    bagDelete(pagesToCrawl);
    hashtable_delete(pagesSeen, deleteItem);
    hashtable_delete(visitedURLs, deleteItem);
}

void pageScan(webpage_t *page, bag_t *pagesToCrawl, hashtable_t *pagesSeen) {
    //initilizing important attributes like the html content and the base url
    char *htmlContent = page->html;
    const char *baseURL = webpageURL(page);
    //Does nothing if its blank
    if (htmlContent == NULL || page->length == 0) {
        return;
    }
    //Initilizing varibles to check for anchor, href, and both single and double quotes
    char *cursor = htmlContent;
    const char *anchor = "<a";
    const char *href = "href=";
    const char *quotes = "\"'"; 
    //Beginning to traverse through the HTML content, with the goal of finding URLS
    while ((cursor = strstr(cursor, anchor)) != NULL) {
        char *hrefStart = strstr(cursor, href);
        //Once an anchor is found it looks for href
        if (hrefStart != NULL) {
            hrefStart += strlen(href);

            //All this logic is essentially handling different quotes (' and ") for the hrefs
            for (int i = 0; quotes[i] != '\0'; ++i) {
                if (*hrefStart == quotes[i]) {
                    char *endURL = strchr(hrefStart + 1, quotes[i]);
                    if (endURL != NULL) {
                        int urlLen = endURL - hrefStart - 1;
                        if (urlLen > 0) {
                            char *foundURL = strndup(hrefStart + 1, urlLen);
                            if (foundURL != NULL) {
                                //Check and skip hashtags (beginning or the end)
                                if (strncmp(foundURL, "#", 1) != 0) {
                                    char *normalizedURL = normalizeURL(baseURL, foundURL);
                                    if (normalizedURL != NULL && isInternalURL(baseURL, normalizedURL)) {
                                        //Adding it to be crawled if never seen before
                                        if (!hashtable_find(pagesSeen, normalizedURL)) {
                                            webpage_t *newPage = (webpage_t *)malloc(sizeof(webpage_t));
                                            if (newPage != NULL) {
                                                newPage->url = normalizedURL;
                                                newPage->html = NULL;
                                                newPage->length = 0;
                                                newPage->depth = page->depth + 1;
					        //Put in the bag to be crawled after making sure the links are all of the correc format	
                                                bagInsert(pagesToCrawl, newPage);

                                                //Inserting the URL into pagesSeen hashtable
                                                unsigned int hash = string_to_hash(normalizedURL) % pagesSeen->num_slots;
                                                set_insert(pagesSeen->set[hash], normalizedURL, normalizedURL);
                                            } else {
						//Freeing all the memory to make sure there is no leaks
                                                free(normalizedURL);
                                                break;
                                            }
                                        } else {
                                            free(normalizedURL);
                                        }
                                    } else {
                                        free(normalizedURL);
                                    }
                                }
                                free(foundURL);
                            }
                            break; //Breaks after processing the URL
                        }
                    }
                }
            }
        }
        //Moving the cursor to check the next upcoming tags
        cursor++;
    }
}

int main(const int argc, char *argv[]) {
    char *seedURL;
    char *pageDirectory;
    int maxDepth;
    //Calling parseArgs to make sure correct values are given
    parseArgs(argc, argv, &seedURL, &pageDirectory, &maxDepth);


    //Initialize the page directory before crawling
    if (!pagedir_init(pageDirectory)) {
	//Prints an error message if directory had a problem
        fprintf(stderr, "Failed to initialize thst page directory.\n");
        return -1;
    }

    //Proceed with crawling the given website
    crawl(seedURL, pageDirectory, maxDepth);

    return 0;
}
