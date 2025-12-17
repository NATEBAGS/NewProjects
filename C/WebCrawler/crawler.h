#ifndef CRAWLER_H
#define CRAWLER_H
#include "curl.h"
#include "url.h"
#include "hashtable.h"
#include "pagedir.h"
#include "set.h"
#include "bag.h"
#include "webpage.h"

bool endsWithHash(const char *str);
void parseArgs(const int argc, char *argv[], char **seedURL, char **pageDirectory, int *maxDepth);
bool isVisited(hashtable_t *visitedURLs, const char *url);
void crawl(char *seedURL, char *pageDirectory, const int maxDepth);
void pageScan(webpage_t *page, bag_t *pagesToCrawl, hashtable_t *pagesSeen);
#endif /* CRAWLER_H */

