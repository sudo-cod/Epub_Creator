import json
import random
import time
import cloudscraper
from lxml import etree

link = ""

def run(page):
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(f'{link}-{page}/').text
    data = {}
    tree = etree.HTML(resp)

    data["title"] = tree.xpath('//header/h1/span/span[@class="chapter-title"]/text()')[0]
    data["content"] = '\n'.join(tree.xpath('//*[@id="novel-content"]//text()'))
    with open(f'{data["title"]}.json', "w",encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
    print(f'{data["title"]} successfully scraped')

if __name__ == '__main__':
    for page in range(3, 149):
        run(page)
        time.sleep(random.randint(1, 5))
