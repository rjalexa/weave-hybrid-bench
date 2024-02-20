""" Reduce Kaggle's Wine Reviews data from 130K to around 20K objects selecting only Italian wines
    Only keep the title as the ID of the wine and the description field
"""

import json

with open("resources/winemag-data-130k-v2.json", "r", encoding="utf-8") as f:
    wine_list_loaded = json.load(f)

print(len(wine_list_loaded))

filtered_wines = [
    {"title": obj["title"], "description": obj["description"]}
    for obj in wine_list_loaded
    if obj["country"] == "Italy"
]
print(len(filtered_wines))

with open("resources/italian-wines-20k-v2.json", "w", encoding="utf-8") as f:
    json.dump(filtered_wines, f, ensure_ascii=False)
