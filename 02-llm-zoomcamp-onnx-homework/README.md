# Homework: Vector Search

_(Homework instructions are verbatim for clarity; I've noted where my code begins to solve the homework problems.)_

In this homework, we put what we learned in Module 2 into practice.

We'll first turn text into vectors, then search by similarity. We'll also learn something new and see how to combine vector search with keyword search. We'll skip the RAG part and focus solely on search.

Like in homework 1, our knowledge base is the course lessons themselves. Each module has a ```lessons/``` folder of numbered markdown pages, and we pull them from GitHub. We use the same commit, ```8c1834d```, so everyone works with the exact same 72 pages.

It's possible your answers won't match exactly. If so, select the closest one.

## Setup
In this homework we won't use the same approach for embedding as in the module. That is, we won't use the sentence-transformers library. Instead, we'll use the lightweight embedding approach with the ONNX ```Embedder```.

Both approaches produce identical vectors, but the ONNX runtime is far lighter. It needs no PyTorch and no CUDA, which makes the installation about 30x smaller and lets it run anywhere, including a basic Codespace. We skimmed through it in the lesson and said we'd cover it in the homework - so here we are.

We prepare the environment the same way as in the module's ONNX Runtime lesson.

Create a fresh project and install the dependencies:

    mkdir llm-zoomcamp-hw2 && cd llm-zoomcamp-hw2
    uv init --no-workspace
    uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
    uv add --dev huggingface-hub jupyter

We also need two helper scripts from the embed/ directory of the course repo:

* ```download.py``` (fetches an ONNX model from HuggingFace) and
* ```embedder.py``` (the Embedder class with an encode interface)

Let's download them:


```python
# PREFIX="https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/02-vector-search/embed"
# !wget $PREFIX/download.py
# !wget $PREFIX/embedder.py
```

By default ```download.py``` fetches ```Xenova/all-MiniLM-L6-v2```, the ONNX version of the ```all-MiniLM-L6-v2``` model from the lessons:


```python
# !uv run python download.py
```

Now we're ready to do the homework.

## Q1. Embedding a query
Embed the following query:

"How does approximate nearest neighbor search work?"


The embedder returns a vector of 384 numbers. What's the first value ```(v[0])```?  The options are:  
* -0.31
* -0.02
* 0.12
* 0.44

_**(My work below**_)


```python
from tqdm.auto import tqdm
import numpy as np
from embedder import Embedder

embed = Embedder()

q1 = "How does approximate nearest neighbor search work?"
v1 = embed.encode(q1)

```


```python
v1.shape
```




    (384,)




```python
v1[0]
```




    np.float64(-0.020582036807885073)



**The first value (v[0]) is -0.02.**

### Loading the data
We pull the lesson pages from the course repository, the same way as in homework 1. We pin to commit ```8c1834d``` so everyone works with the same data.


```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
```

Each document is a dictionary with a ```filename``` and ```content```, and there are 72 pages.


```python
# Let's double-check the size of the documents file to confirm
len(documents)
```




    72



_**(My work below)**_

Note:  The codeblock below limits the length of the results text displayed in the notebook, to make it easier to read when scrolling through the notebook.  The command ```show(document)``` activates this mode.  


```python
import re

def show(obj, max_items=20, max_chars_per_item=250):
    """Display object showing first N items, each truncated to max chars"""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # List of dictionaries
        output = "["
        for i, item in enumerate(obj[:max_items]):
            if i > 0:
                output += "\n"
            output += "{"
            keys = list(item.keys())
            for j, key in enumerate(keys):
                value = item[key]
                key_str = str(key)
                value_str = str(value)
                
                value_str = re.sub(r'[\n\r\t]+', '', value_str)
                
                if len(key_str) > max_chars_per_item:
                    key_str = key_str[:max_chars_per_item] + "..."
                if len(value_str) > max_chars_per_item:
                    value_str = value_str[:max_chars_per_item] + "..."
                
                comma = "," if j < len(keys) - 1 else ""
                if j == 0:
                    output += f"'{key_str}': '{value_str}'{comma}"
                else:
                    output += f"\n  '{key_str}': '{value_str}'{comma}"
            
            comma = "," if i < len(obj[:max_items]) - 1 else ""
            output += f"}}{comma}"
        
        output += "]"
        print(output)
        if len(obj) > max_items:
            print(f"(and {len(obj) - max_items} more items)")
    
    elif isinstance(obj, list):
        # Regular list of strings/items
        output = "["
        for i, item in enumerate(obj[:max_items]):
            if i > 0:
                output += "\n "
            item_str = re.sub(r'[\n\r\t]+', '', str(item))
            if len(item_str) > max_chars_per_item:
                item_str = item_str[:max_chars_per_item] + "..."
            comma = "," if i < len(obj[:max_items]) - 1 else ""
            output += f"'{item_str}'{comma}"
        
        output += "]"
        print(output)
        if len(obj) > max_items:
            print(f"(and {len(obj) - max_items} more items)")
```


```python
# Let's look at the first two documents:
show(documents[:2])
```

    [{'content': '# IntroductionVideo: [Watch this lesson](https://www.youtube.com/watch?v=rQYyFxf1FWw&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)In this module, we'll build a working Retrieval-AugmentedGeneration (RAG) s...',
      'filename': '01-agentic-rag/lessons/01-intro.md'},
    {'content': '# EnvironmentVideo: [Watch this lesson](https://www.youtube.com/watch?v=3U4gBrmkZyM&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)For this module, all you need is Python with Jupyter.## PrerequisitesYou nee...',
      'filename': '01-agentic-rag/lessons/02-environment.md'}]


## Q2. Cosine similarity
The embedder returns normalized vectors, so the dot product between two of them is their cosine similarity.

Take the page ```02-vector-search/lessons/07-sqlitesearch-vector.md```, embed its ```content```, and compute the cosine similarity with the query vector from Q1. What do you get?

* 0.07
* 0.37
* 0.68
* 0.92


_**(My work below)**_

First, I need to turn the python dictionaries into straight text to perform embedding.


```python
texts = [doc["content"] + " " + doc["filename"] for doc in documents]
```

Now, I'll search the texts file for the course page:



```python
target = "02-vector-search/lessons/07-sqlitesearch-vector.md"

result = None
for item in texts:
    if target in item:
        result = item
        break

if result:
    print("Found!")
else:
    print("Not found")
```

    Found!



```python
len(result)
```




    7270



Now I need to embed the content for this lesson page and compute the cosine similarity with the question vector from Q1.


```python
v_result = embed.encode(result)
```


```python
v_result.shape
```




    (384,)



To get similarity score, multiply the matrices:  


```python
v1.dot(v_result)
```




    np.float64(0.361070280302606)



**The score, 0.36, is closest to the multiple choice option of 0.37.**

## Q3. Chunking and search by hand
A full page covers several topics, which waters down its embedding.

We chunk the pages the same way as in homework 1:

    from gitsource import chunk_documents
    chunks = chunk_documents(documents, size=2000, step=1000)

We embed every chunk's content with ```encode_batch```, stack the vectors into a matrix ```X```, and score the Q1 query against all chunks:

    scores = X.dot(v)

Which file does the highest-scoring chunk belong to (its filename)?

* 02-vector-search/lessons/03-embeddings-dataset.md
* 02-vector-search/lessons/06-rag-vector.md
* 02-vector-search/lessons/07-sqlitesearch-vector.md
* 02-vector-search/lessons/09-onnx-embedder.md


```python
from gitsource import chunk_documents
chunks = chunk_documents(documents, size=2000, step=1000)
```

_**(My work below)**_

Checking the result of chunking:


```python
len(chunks)
```




    295




```python
show(chunks[:3])
```

    [{'start': '0',
      'content': '# IntroductionVideo: [Watch this lesson](https://www.youtube.com/watch?v=rQYyFxf1FWw&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)In this module, we'll build a working Retrieval-AugmentedGeneration (RAG) system from scratch, step by step.We write everythi...',
      'filename': '01-agentic-rag/lessons/01-intro.md'},
    {'start': '1000',
      'content': 'the nextword based on what you typed so far.A large language model does the same thing, but at a much larger scale.It has billions of parameters and is trained on most of the text on theinternet. When it predicts the next word, it feels like you're t...',
      'filename': '01-agentic-rag/lessons/01-intro.md'},
    {'start': '2000',
      'content': 'wrong.## The projectRAG solves these problems by giving the LLM relevant documents atquestion time. We don't hope the model memorized the answer. Weretrieve the right information and hand it to the LLM, and the modelgenerates a grounded response. Thi...',
      'filename': '01-agentic-rag/lessons/01-intro.md'}]



```python
# Build one text per chunk; put them all together to get the texts corpus

chunks_texts = []

for chunk in chunks:
    chunk_text = chunk["filename"] + " " + chunk["content"]
    chunks_texts.append(chunk_text)
```


```python
show(chunks_texts[:3])
```

    ['01-agentic-rag/lessons/01-intro.md # IntroductionVideo: [Watch this lesson](https://www.youtube.com/watch?v=rQYyFxf1FWw&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)In this module, we'll build a working Re...',
     '01-agentic-rag/lessons/01-intro.md the nextword based on what you typed so far.A large language model does the same thing, but at a much larger scale.It has billions of parameters and is trained on mo...',
     '01-agentic-rag/lessons/01-intro.md wrong.## The projectRAG solves these problems by giving the LLM relevant documents atquestion time. We don't hope the model memorized the answer. Weretrieve the righ...']


We embed every chunk's ```content``` with ```encode_batch```, stack the vectors into a matrix ```X```, and score the Q1 query against all chunks:


```python
embed = Embedder()

batch_size = 50
X = []

for i in tqdm(range(0, len(chunks_texts), batch_size)):
    batch = chunks_texts[i:i + batch_size]
    batch_vectors = embed.encode_batch(batch)
    X.extend(batch_vectors)

X = np.array(X)
```


      0%|          | 0/6 [00:00<?, ?it/s]



```python
X
```




    array([[-0.09669086,  0.05101552, -0.05132843, ...,  0.05558655,
            -0.03937108,  0.03190763],
           [ 0.05609387, -0.09380053,  0.00662726, ...,  0.063732  ,
            -0.03171691,  0.02321687],
           [-0.01367305,  0.0212395 , -0.0218254 , ...,  0.06771878,
            -0.02497036,  0.03221907],
           ...,
           [-0.02140815,  0.04564691,  0.02515901, ..., -0.04789668,
            -0.08538731,  0.04617506],
           [-0.0467181 ,  0.03808443, -0.00793749, ...,  0.04502728,
             0.01948824,  0.0478768 ],
           [-0.04896879, -0.01691668, -0.01808992, ...,  0.00830904,
             0.01757162,  0.01088019]], shape=(295, 384))




```python
X.shape
```




    (295, 384)




```python
q = "How does approximate nearest neighbor search work?"
v = embed.encode(q)
```

Getting scores for how well each document matches the question:


```python
scores = X.dot(v)
```

Get the index of the chunk with the highest score:


```python
idx = np.argmax(scores)
idx
```




    np.int64(94)



Checking the chunk with the highest score by selecting that index value from the chunk_texts (the list of documents in text form):


```python
chunks_texts[94]
```




    '02-vector-search/lessons/07-sqlitesearch-vector.md rch. We score\nthe query against every document and pick the top ones. It always finds\nthe true top matches, but it pays for that by touching everything.\n\nApproximate nearest neighbor (ANN) search takes a shortcut. Instead of\ncomparing against everything, it first narrows down to a region of\nlikely matches. Then it scores only within that region. It may miss the\nabsolute best match, but the results are still good and it\'s much\nfaster.\n\n```text\nNN (exact):    compare query against ALL documents -> top 5\nANN (approx):  narrow down to a region -> compare within region -> top 5\n```\n\n## sqlitesearch\n\nsqlitesearch is the persistent sibling of minsearch, and it solves both\nproblems at once.\n\nWe already used it in module 1 for persistent text search. It also does\nvector search through its `VectorSearchIndex` class. It stores vectors\nin SQLite, a real on-disk database, and uses ANN strategies for\nretrieval. Because the data lives on disk, one process can write the\nvectors and another can read them back.\n\nIf you didn\'t install it in the previous module, add it to your project:\n\n```bash\nuv add sqlitesearch\n```\n\n## Creating the index\n\nInitialize it:\n\n```python\nfrom sqlitesearch import VectorSearchIndex\n\nvs_index = VectorSearchIndex(\n    keyword_fields=["course"],\n    mode="ivf",\n    db_path="faq_vectors2.db"\n)\n```\n\nsqlitesearch supports three ANN modes:\n\n- `lsh` (default): up to 100K vectors, random hyperplane projections\n- `ivf`: 10K-500K vectors, K-means clustering\n- `hnsw`: 10K-1M+ vectors, proximity graph (highest recall)\n\nFor our small dataset, `lsh` is fine. All modes use two-phase search:\napproximate candidate retrieval, then exact cosine similarity\nreranking.\n\n## Indexing the data\n\nFit the index with our vectors and documents:\n\n```python\nvs_index.fit(vectors, documents)\n```\n\nThe index is saved to `faq_vectors2.db`. Unlike minsearch, this file\npersists on disk. You can search immediately after indexing, or reopen\nthe index later without re-indexing.\n\n## Searching\n\nSearch '




```python
# Checking the score at index 94

scores[94]
```




    np.float64(0.5685932153175144)



**The highest scoring chunk is at index 94 and has a score of 0.57.  The filename is ```02-vector-search/lessons/07-sqlitesearch-vector.md```.**

## Q4. Vector search with minsearch
We've done vector search by hand, which is good for learning, but it's not what we do in practice. In practice we use libraries.

Let's use ```VectorSearch``` from minsearch and run a search for the following query:

"What metric do we use to evaluate a search engine?"

Which file is the ```filename``` of the first result?

* 02-vector-search/lessons/04-vector-search.md
* 04-evaluation/lessons/05-search-metrics.md
* 04-evaluation/lessons/13-llm-as-judge.md
* 05-monitoring/lessons/04-metrics.md

_**(Code below is mine)**_

First, I need to embed the texts and also the question being asked.  



```python
from minsearch import VectorSearch

vindex = VectorSearch()
vindex.fit(X, chunks)
```




    <minsearch.vector.VectorSearch at 0x17efce210>




```python
query = "What metric do we use to evaluate a search engine?"
query_embedded = embed.encode(query)
```

Get the results of the vector search for the question.  Note that minsearch vector search ranks the results, with the first result corresponding to the highest score.


```python
results = vindex.search(query_embedded, num_results=5)
```


```python
show(results[0:3])
```

    [{'start': '0',
      'content': '# Search Evaluation MetricsVideo: [Watch this lesson](https://www.youtube.com/watch?v=TuirMy3Pdbk&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)In the previous lesson, we computed relevance lists for search results.We can turn those lists into metrics.## H...',
      'filename': '04-evaluation/lessons/05-search-metrics.md'},
    {'start': '0',
      'content': '# EvaluationVideo: [Watch this lesson](https://www.youtube.com/watch?v=eC_IcxfxoiQ&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)In the previous modules, we built search engines and RAG pipelines.We tried different approaches: keyword search with minsearch...',
      'filename': '04-evaluation/lessons/01-intro.md'},
    {'start': '2000',
      'content': ' Offline evaluation: run the system on a test dataset and compute metrics- Online evaluation: collect feedback from real users in productionOffline evaluation is what we do before putting changes in front ofusers. It lets us compare search settings, ...',
      'filename': '04-evaluation/lessons/01-intro.md'}]


**The first result is ```04-evaluation/lessons/05-search-metrics.md```.** 

## Q5. Text search vs vector search
Vector search matches by meaning, keyword search by exact words.

Let's compare them. Index the same chunks with ```Index``` from minsearch. Use ```content``` as a text field.

Run both searches for this query:

"How do I store vectors in PostgreSQL?"

Take the top 5 results from each method. Which file shows up in the vector results but not in the text results?

* 02-vector-search/lessons/01-intro.md
* 02-vector-search/lessons/02-embeddings.md
* 02-vector-search/lessons/08-pgvector.md
* 03-orchestration/lessons/05-rag.md

_**(My work follows)**_

I'll perform the text search first, using the ingest.py file from an earlier lesson:


```python
# from ingest.py

import requests
from minsearch import Index

def build_index(documents):
    index = Index(
        text_fields=['content'],
    )
    index.fit(documents)
    return index
```

Here I create the index from the chunks of lesson documents imported earlier.


```python
index = build_index(chunks)
```

Executing the search:


```python
# Because it's text search, not vector search, we don't need to embed the query into a vector 

query = "How do I store vectors in PostgreSQL?"

search_results = index.search(
    query,
    num_results=5
)

show(search_results)
```

    [{'start': '4000',
      'content': 'get 0.01.The first score for `q1` vs `d` (0.32) is higher, so that query is moresimilar to the document about registration. The second score for `q2`vs `d` sits near 0, because installing Docker has nothing to do withregistration. A score near 0 mean...',
      'filename': '02-vector-search/lessons/02-embeddings.md'},
    {'start': '1000',
      'content': 'arch](../../02-vector-search/lessons/04-vector-search.md).## How RAG Works in KestraRAG has two phases. In the demo flows below they run back-to-back, but in production you'd typically schedule them separately — ingest on a cadence, query on demand.`...',
      'filename': '03-orchestration/lessons/05-rag.md'},
    {'start': '0',
      'content': '# Vector SearchVideo: [Watch this lesson](https://www.youtube.com/watch?v=qyZgxTmC2cY&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)In module 1 we used keyword search with minsearch and sqlitesearch.It matches exact words. If you search for "Docker", the d...',
      'filename': '02-vector-search/lessons/01-intro.md'},
    {'start': '0',
      'content': '# Retrieval Augmented GenerationVideo: [RAG Workflows](https://youtu.be/FhGZV173xrk)AI Copilot solves the context problem for flow generation. But what about workflows that need to answer questions from your own data? That's where RAG comes in.> Note...',
      'filename': '03-orchestration/lessons/05-rag.md'},
    {'start': '1000',
      'content': 'dding model produces these vectors. It's a neural networktrained to capture meaning, so texts that mean similar things land onsimilar vectors. We measure how close two vectors are with a distancemetric. The most common one is cosine similarity.Cosine...',
      'filename': '02-vector-search/lessons/01-intro.md'}]


Now I'll perform the same search with minserch VectorSearch.  I only need to embed the query before running the search because I already have the vector embeddings for the chunks of text. 


```python
query = "How do I store vectors in PostgreSQL?"
query_embedded = embed.encode(query)
```


```python
results = vindex.search(query_embedded, num_results=5)
```


```python
show(results)
```

    [{'start': '0',
      'content': '# Vector Search with PGVectorVideo: [Watch this lesson](https://www.youtube.com/watch?v=0P54MFyz-mc&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)Many real databases can do vector search. Elasticsearch has it, andthere are dedicated stores like Qdrant and ...',
      'filename': '02-vector-search/lessons/08-pgvector.md'},
    {'start': '3000',
      'content': 'PGVector:```pythondef vec_to_str(vector):    return "[" + ",".join(str(x) for x in vector) + "]"for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):    conn.execute(        """        INSERT INTO documents (course, section, question, ...',
      'filename': '02-vector-search/lessons/08-pgvector.md'},
    {'start': '4000',
      'content': 'n, answer,           1 - (embedding <=> %s::vector) AS similarity    FROM documents    ORDER BY embedding <=> %s::vector    LIMIT 5    """,    (query_str, query_str)).fetchall()for row in results:    print(f"[{row[0]}] {row[1]} (similarity: {row[3]:....',
      'filename': '02-vector-search/lessons/08-pgvector.md'},
    {'start': '1000',
      'content': 'ar/lib/postgresql/data \    -p 5432:5432 \    pgvector/pgvector:pg17```This image has the pgvector extension pre-installed. The `-v` flagcreates a named volume so data persists across container restarts.## Installing the Python clientInstall the driv...',
      'filename': '02-vector-search/lessons/08-pgvector.md'},
    {'start': '2000',
      'content': ' = model.encode(batch)    vectors.extend(batch_vectors)```Now we connect to Postgres:```pythonimport psycopgconn = psycopg.connect(    "postgresql://user:pswd@localhost:5432/faq")conn.execute("CREATE EXTENSION IF NOT EXISTS vector")```The second line...',
      'filename': '02-vector-search/lessons/08-pgvector.md'}]


_**The result that shows up in vector search but does not show up in text search is ```02-vector-search/lessons/08-pgvector.md```.**_

## Q6. Hybrid search
Both vector and text search have their strengths and weaknesses. Vector search matches by meaning, so it finds relevant pages even when they use words different from the query. But it can miss exact terms like names, codes, or rare keywords. Text search is the opposite: it nails exact words but misses paraphrases and synonyms.

We don't have to pick one or the other - we can use both and merge their results. This approach is called "hybrid search".

Each search produces its own ranked list, so we need a way to combine them into one. In this homework we use Reciprocal Rank Fusion (RRF). It ignores the raw scores from each method, which live on different scales and aren't directly comparable. Instead, it looks only at the position of each document in each list.

Every document scores by its position (```rank```, starting at 0) in each list, and we sum the scores across lists with a constant ```k = 60```:

```RRF(d) = sum over lists of  1 / (k + rank(d))```

"Sum over lists" means we go through every ranked list and, for each list where the document appears, add its ```1 / (k + rank)``` contribution. A document found by both searches collects a score from each list, while one found by only a single search collects just one.

The constant ```k``` controls how much the exact rank matters. A larger ```k``` flattens the gap between positions, so the difference between rank 0 and rank 5 counts for less. A smaller ```k``` does the opposite: it sharpens that gap, so being at the top of a list matters much more.

The value 60 comes from the original RRF paper and is the usual default. You rarely need to tune it. Lower it when only the top results matter. Raise it to reward documents that appear across many lists, even when they never quite reach the top.

A document that ranks well in both lists ends up higher than one that's only strong in a single list.

Which file is ranked first after running the RRF function?

* 01-agentic-rag/lessons/01-intro.md
* 01-agentic-rag/lessons/13-function-calling.md
* 01-agentic-rag/lessons/14-agentic-loop.md
* 01-agentic-rag/lessons/16-other-frameworks.md

Notice that this file isn't first in either search on its own - it wins because it ranks high in both.




```python
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)   
    return [docs[key] for key in ranked[:num_results]]
```

Now run the query ```"How do I give the model access to tools?"``` with vector and text search and fuse the results with ```rrf```.

_**(My work is below)**_

First, I'll set up the search using minsearch text search.  


```python
# Using text search

query = "How do I give the model access to tools?"

text_results = index.search(query)

print(f"Total number of text search results: {len(text_results)}")
show(text_results[:5])

```

    Total number of text search results: 10
    [{'start': '0',
      'content': '# The Agentic LoopVideo: [Watch this lesson](https://www.youtube.com/watch?v=ePlQUcTPPjw&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)In the previous lesson, we did function calling by hand. We sent amessage and got back a function call. We ran it, sent t...',
      'filename': '01-agentic-rag/lessons/14-agentic-loop.md'},
    {'start': '4000',
      'content': ' function. `parameters` is a JSON schemafor the arguments, and we mark `query` as required so the model alwaysfills it in.## Sending the question with the toolNow we send the same question as before, but this time we include thetool in the request:``...',
      'filename': '01-agentic-rag/lessons/13-function-calling.md'},
    {'start': '5000',
      'content': 'unction, and serialize the result.```pythonimport jsoncall = response.output[0]args = json.loads(call.arguments)results = search(**args)result_json = json.dumps(results, indent=2)```Now we send this result back to the model. First, we add the model's...',
      'filename': '01-agentic-rag/lessons/13-function-calling.md'},
    {'start': '1000',
      'content': 'decides when to call it and what to search for.The same typo question now goes like this:```mermaidflowchart TD    U([User: How do I run Olama?])    L1[LLM: I'll search for 'Olama']    S1[search - Olama - no useful results]    L2[LLM: Hmm, no results...',
      'filename': '01-agentic-rag/lessons/13-function-calling.md'},
    {'start': '3000',
      'content': 'the output. The model returns thesame structure every time. We can access the generated questionsdirectly instead of parsing text manually.We want the output as a list of strings, so we define that structurewith a Pydantic model:```pythonfrom pydanti...',
      'filename': '04-evaluation/lessons/02-ground-truth.md'}]



```python
text_scores = X.dot(query_embedded)
print(f"There are {len(text_scores)} scores; printing the first 20 scores:")
text_scores[:20]
```

    There are 295 scores; printing the first 20 scores:





    array([ 0.08054137, -0.0141192 , -0.0593237 , -0.04490812, -0.1145863 ,
            0.07918184, -0.0374468 , -0.048944  , -0.12243607, -0.06149635,
           -0.00490393,  0.03213537, -0.07563486,  0.05773673,  0.01932183,
            0.03547798, -0.04588024,  0.07453409, -0.01284978, -0.00225151])




```python
# Using vector search
query_embedded = embed.encode(query)

vector_results = vindex.search(query_embedded)

print(f"Number of vector search results: {len(vector_results)}; printing the first 5 results:")
show(vector_results[:5])
```

    Number of vector search results: 10; printing the first 5 results:
    [{'start': '2000',
      'content': ' num_results=5,        boost_dict={"question": 3.0, "section": 0.5},        filter_dict={"course": "llm-zoomcamp"}    )```Then register it without passing a schema:```pythonagent_tools = Tools()agent_tools.add_tool(search)```You can look at what ToyA...',
      'filename': '01-agentic-rag/lessons/15-frameworks.md'},
    {'start': '1000',
      'content': 'swer.## Loading the documentsWe'll use helper files from module 01 and this module.If you don't have them in your notebook directory, download them:```bashPREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/mainwget ${PREFIX}/01-agent...',
      'filename': '04-evaluation/lessons/02-ground-truth.md'},
    {'start': '0',
      'content': '# Other FrameworksVideo: [Watch this lesson](https://www.youtube.com/watch?v=4yiCbKX9RhI&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)The concepts you learned in Part 2 are the same across frameworks.Function calling, the agent loop, and tool definitions ...',
      'filename': '01-agentic-rag/lessons/16-other-frameworks.md'},
    {'start': '4000',
      'content': ' function. `parameters` is a JSON schemafor the arguments, and we mark `query` as required so the model alwaysfills it in.## Sending the question with the toolNow we send the same question as before, but this time we include thetool in the request:``...',
      'filename': '01-agentic-rag/lessons/13-function-calling.md'},
    {'start': '2000',
      'content': ' see what the LLM does without any tools. We ask it acourse-specific question and look at the answer.```pythonmessages = [    {"role": "user", "content": "I just discovered the course. Can I join it?"}]response = openai_client.responses.create(    mo...',
      'filename': '01-agentic-rag/lessons/13-function-calling.md'}]



```python
results = rrf([vector_results, text_results])
```


```python
show(results)
```

    [{'start': '4000',
      'content': ' function. `parameters` is a JSON schemafor the arguments, and we mark `query` as required so the model alwaysfills it in.## Sending the question with the toolNow we send the same question as before, but this time we include thetool in the request:``...',
      'filename': '01-agentic-rag/lessons/13-function-calling.md'},
    {'start': '5000',
      'content': 'unction, and serialize the result.```pythonimport jsoncall = response.output[0]args = json.loads(call.arguments)results = search(**args)result_json = json.dumps(results, indent=2)```Now we send this result back to the model. First, we add the model's...',
      'filename': '01-agentic-rag/lessons/13-function-calling.md'},
    {'start': '2000',
      'content': ' see what the LLM does without any tools. We ask it acourse-specific question and look at the answer.```pythonmessages = [    {"role": "user", "content": "I just discovered the course. Can I join it?"}]response = openai_client.responses.create(    mo...',
      'filename': '01-agentic-rag/lessons/13-function-calling.md'},
    {'start': '2000',
      'content': ' num_results=5,        boost_dict={"question": 3.0, "section": 0.5},        filter_dict={"course": "llm-zoomcamp"}    )```Then register it without passing a schema:```pythonagent_tools = Tools()agent_tools.add_tool(search)```You can look at what ToyA...',
      'filename': '01-agentic-rag/lessons/15-frameworks.md'},
    {'start': '0',
      'content': '# The Agentic LoopVideo: [Watch this lesson](https://www.youtube.com/watch?v=ePlQUcTPPjw&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)In the previous lesson, we did function calling by hand. We sent amessage and got back a function call. We ran it, sent t...',
      'filename': '01-agentic-rag/lessons/14-agentic-loop.md'}]


_**The document that ranks highest in RRF is ```01-agentic-rag/lessons/13-function-calling.md```.  It appears on both lists, but is not the highest-ranked document on either list.**_ 
