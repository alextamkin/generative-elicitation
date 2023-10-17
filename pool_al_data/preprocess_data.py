import csv
from urllib.parse import urlparse
import json
import random

# get domain name of url
def get_domain(url):
    return url.split("//")[-1].split("/")[0]

csv_reader = csv.DictReader(open("pool_al_data/MINDsmall_train/news.tsv", 'r'), delimiter='\t')
websites = []
categories_to_subcategories_to_websites = {}
with open("pool_al_data/website_preferences.jsonl", 'w') as wf:
    for row in csv_reader:
        domain = urlparse(row['URL']).netloc
        domain = '.'.join(domain.split('.')[-2:])
        if domain != "" and domain != "msn.com":
            assert False
        website_description = f"Website: {domain}\nTitle: {row['Title']}\nDescription: {row['Abstract']}"
        websites.append(website_description)
        wf.write(json.dumps({"nl_desc": website_description, "full_entry": row})+"\n")

        if row['Category'] not in categories_to_subcategories_to_websites:
            categories_to_subcategories_to_websites[row['Category']] = {}
        if row['SubCategory'] not in categories_to_subcategories_to_websites[row['Category']]:
            categories_to_subcategories_to_websites[row['Category']][row['SubCategory']] = []
        categories_to_subcategories_to_websites[row['Category']][row['SubCategory']].append((website_description, row))

num_categories = len(categories_to_subcategories_to_websites)
num_subcategories = sum([len(categories_to_subcategories_to_websites[category]) for category in categories_to_subcategories_to_websites])
# get a diverse subset of websites
with open("pool_al_data/website_preferences_subsample.jsonl", 'w') as wf:
    for category in categories_to_subcategories_to_websites:
        for subcategory in categories_to_subcategories_to_websites[category]:
            website_description = random.choice(categories_to_subcategories_to_websites[category][subcategory])
            wf.write(json.dumps({"nl_desc": website_description[0], "full_entry": website_description[1]})+"\n")

