from bs4 import BeautifulSoup
import re

import datetime
import pandas as pd
import os


def ParseDate(date):
    monthDict = {
        'januari' : 1,
        'februari': 2,
        'mars': 3,
        'april': 4,
        'maj': 5,
        'juni': 6,
        'juli': 7,
        'augusti': 8,
        'september': 9,
        'oktober': 10,
        'november': 11,
        'december': 12
    }
    parts = date.split(" ")

    return datetime.datetime(day=int(parts[1]), month=monthDict[parts[2]], year=int(parts[3]))

def Process(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')

    cols = 7
    data = [[]*cols for i in range(cols)]
    for listing in soup.find_all ('li', class_ ='sold-results__normal-hit'):
        data[0].append(ParseDate(listing.find('span', class_ ='hcl-label--sold-at').text.strip()))
        data[1].append(listing.find('h2', class_='sold-property-listing__heading').text.strip())
        # Get raw location
        location = listing.select('div.sold-property-listing__location>div')[0].text.strip()
        location = re.sub(r'VillaVilla\s+|\n|  ', '', location) # Remove garbage
        data[2].append(location)

        area = listing.select('div.sold-property-listing__area')[0].text.strip()
        text = re.sub(r'\s+', '', area).split('²')
        rooms = re.sub(r'rum', '', text[len(text)-1])
        area = 0
        for i in range(len(text)-1):
            for temp in text[i].split('+'):
                area += float(re.sub(',+', '.', re.sub('m+|m²+', '', temp)))

        data[3].append(area)
        data[4].append(rooms)
        
        if(listing.find('div', class_ ='sold-property-listing__land-area')):
            landarea = listing.select('div.sold-property-listing__land-area')[0].text.strip()
            landarea = re.sub(r'm² tomt|\s+', '', landarea) # Remove garbage
            data[5].append(landarea)
        else:
            data[5].append(0) # Land area doesnt exist, maybe an apartment?

        price = re.sub(r'\s+|Slutpris|kr', '', listing.find('span', 'hcl-text').text.strip())
        data[6].append(price)

    df = pd.DataFrame({
        'date': data[0],
        'address': data[1],
        'location': data[2],
        'boarea': data[3],
        'rooms': data[4],
        'landarea': data[5],
        'closing price': data[6],
    })
    return df

folder_path = 'kungalv_slutpriser'
files = os.listdir (folder_path)

table = list()

# For each filename
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    f = open(file_path, 'r', encoding='utf-8')
    table.append(Process(f.read()))

df = pd.concat(table)

df.to_csv('listings.csv', index = None)


