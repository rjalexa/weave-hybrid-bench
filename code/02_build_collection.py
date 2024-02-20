""" 
    Define a simple Weaviate collection with onjects that have just two properties (fields)
        both are text, the title is not tokenized (field), the description is word tokenized
        the collection will use english stopwords for the inverted index
        we are going to use Openai to let Weaviate automatically vectorize new objects and queries
    Read the reduced italian wine data and inject it into its weaviate collection 
"""

import os
import json
from dotenv import load_dotenv
import weaviate
import weaviate.classes.config as wvcc

load_dotenv()
schema_name = os.getenv("COLLNAME")
openai_key = os.getenv("API_KEY")
whost = os.getenv("WHOST")
wport = os.getenv("WPORT")
oi_ratelimit = int(os.getenv("OPENAI_RATELIMIT"))


with weaviate.connect_to_local(  # this will connect and then at the end implicitely close
    host=whost,
    port=wport,
    headers={
        "X-OpenAI-Api-Key": openai_key,  #  for generative queries
    },
) as client:
    if client.collections.exists(schema_name):
        client.collections.delete(schema_name)
    client.collections.create(
        schema_name,
        description="A class to store wine names and their descriptions",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvcc.Configure.Generative.openai(),
        inverted_index_config=wvcc.Configure.inverted_index(
            stopwords_preset=wvcc.StopwordsPreset(
                "en"
            )  # since the descriptions are written in English ...
        ),
        vector_index_config=wvcc.Configure.VectorIndex.hnsw(
            distance_metric=wvcc.VectorDistances.COSINE
        ),
        properties=[
            # default tokenization is tokenization=wvcc.Tokenization.WORD
            wvcc.Property(
                name="title",
                data_type=wvcc.DataType.TEXT,
                tokenization=wvcc.Tokenization.FIELD,
            ),
            wvcc.Property(name="description", data_type=wvcc.DataType.TEXT),
        ],
    )
print(f"Successfully created a new empty {schema_name} schema.")
# client is implicitely close because of the "with" context

# now fetch the data
with open("resources/italian-wines-20k-v2.json", "r", encoding="utf-8") as f:
    ita_wine_list = json.load(f)
print(f"Loaded {len(ita_wine_list)} JSON onjects from the file.")

# store the wine data in the wine collection
client = weaviate.connect_to_local(
    host=whost,
    port=wport,
    headers={
        "X-OpenAI-Api-Key": openai_key,  #  for generative queries
    },
)
if client.is_ready():
    try:
        wines_coll = client.collections.get(schema_name)
        with wines_coll.batch.rate_limit(requests_per_minute=oi_ratelimit) as batch:
            num = 1
            for wine_data in ita_wine_list:
                print(f"Inserting object #:{num:05}")
                batch.add_object(
                    properties=wine_data,
                )
                num += 1
        response = wines_coll.aggregate.over_all(total_count=True)
        print(f"We now have {response.total_count} in the {schema_name} collection")
        print("Failed Objects", wines_coll.batch.failed_objects)
        print("Failed References", wines_coll.batch.failed_references)
    finally:
        client.close()
