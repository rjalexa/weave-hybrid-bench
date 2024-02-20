""" Select N random elements from the Italian wines list. Ensure they are always the same ones 
   NB: I have chosen to use Stanford's Stanza NLP modules to generate the keywords 
   but you are free to use whatever model and strategy to generate your own from the description
"""

import random
import json
import os
from dotenv import load_dotenv
import stanza

with open("resources/italian-wines-20k-v2.json", "r", encoding="utf-8") as f:
    ita_wine_list = json.load(f)
print(f"Loaded {len(ita_wine_list)} JSON onjects from the file.")

load_dotenv()
benchmark_size = int(os.getenv("BENCH_SIZE"))

# Set a fixed seed for the random number generator
random.seed(42)

# Randomly select N elements from the list
selected_wines = random.sample(ita_wine_list, benchmark_size)

keywords_benchmark = []

stanza.download("en")
nlp = stanza.Pipeline("en", processors="tokenize,pos")

for one_wine in selected_wines:
    doc = nlp(one_wine["description"])
    nouns = []
    verbs = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == "NOUN":  # Check if the word is a noun
                nouns.append(word.text)
            elif word.upos == "VERB":  # Check if the word is a verb
                verbs.append(word.text)
    keywords = nouns + verbs
    keywords_benchmark.append({"title": one_wine["title"], "keywords": keywords})

# write the benchmark to file
benchfn = f"resources/italian-wines-{benchmark_size}-keywords-bench.json"

with open(benchfn, "w", encoding="utf-8") as f:
    json.dump(keywords_benchmark, f, ensure_ascii=False)

print(f"Wrote {benchmark_size} benchmark objects with keywords to file {benchfn}")
