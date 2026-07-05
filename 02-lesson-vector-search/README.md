# Module 2:  Vector Search
In this module, we implement vector search, which is a more robust form of search that matches documents based on meaning rather than just matching exact keywords in a query with keywords in documents.  

## What is vector space?  
The reason why it's call vector search is because the text is turned into vectors (a series of numbers representing the text).  This is also known as "embedding" the text in a vector space.  

### Word embeddings and sentence embeddings
A word's vector represents it at a point in a multi-dimensional space.  Words with similar meanings are closer together in this multidimensional space, while those whose meanings are dissimilar are further apart.  

Likewise for sentences:  an entire sentence can be embedded as a vector in multi-dimensional space as a single point, not just as a bunch of words in isolation.  This does a better job of representing the meaning of that sentence than the "bag of words" approach used in simpler search methodologies.  

In the case of our FAQ, each question and answer (each "document") is embedded in vector space.  When we ask a question, that question is embedded as a vector into that same multi-dimensional space.  The model finds the documents closest in space to the question (its nearest neighbors) and produces these as the search results.  

## The vector search process 
_(The text below is verbatim from the lesson because it lays out what we're going to be doing in each lesson section.  If the text in a section is not designated as verbatim from the lesson section, it's my own commentary and learning summary.)_

We run vector search in two stages.

* Offline (indexing): we convert all documents into vectors (arrays of numbers) and store them in an index.
* Online (querying): we convert the user's query into a vector with the same model, then find the closest document vectors by similarity.

An embedding model produces these vectors. It's a neural network trained to capture meaning, so texts that mean similar things land on similar vectors. We measure how close two vectors are with a distance metric. The most common one is cosine similarity.

Cosine similarity measures the angle between two vectors:

* Vectors pointing in the same direction: similarity close to 1 (similar)
* Vectors at right angles: similarity close to 0 (unrelated)
* Vectors pointing in opposite directions: similarity close to -1 (opposite meaning)

The larger the cosine similarity, the more similar the two texts are in meaning.

### Building vector search
We'll take the same FAQ dataset from module 1 and build vector search with three tools:

1. ```minsearch``` - in-memory vector search (simplest, good for experiments)
2. ```sqlitesearch``` - persistent vector search backed by SQLite (production-friendly, same API as minsearch)
3. ```PGVector``` - vector search in PostgreSQL (scalable, runs in Docker)

Then we'll plug vector search into our RAG pipeline.

### Introduction to Vector Search using SBERT

To begin, we're going to be using Sentence Transformers (SBERT) a Python module for using and training state-of-the-art embedding and reranker models.  Sentence Transformers was created by UKP Lab and is being maintained by 🤗 Hugging Face.  

I've installed SBERT in my environment, so I can start using this model in the lesson.  


```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2') 
```

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]


Here are a couple questions we'll use in this process:


```python
q1 = "I just discovered the course, can I still join?"
q2 = "I just found out about the program, can I still enroll?"
```


```python
v1 = model.encode(q1)
```

We've produced a vector by encoding the question in the model.  

Let's see the shape of the vector:


```python
v1.shape
```




    (384,)



The vector has 384 values.

Let's encode the second question:


```python
v2 = model.encode(q2)
```


```python
v2.shape
```




    (384,)



New question, new encoding:


```python
q1 = 'Can I still join the course after the start date?'
v1 = model.encode(q1)
```

As with the questions, we'll encode the document (answer) as a vector:


```python
d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
dv = model.encode(d)
```

To find out the similarity between the question and the document (answer), we perform vector multiplication.  The higher the score, the better the match.  A score close to zero indicate that the model has found little or no similarity between the two vectors, whereas a score closer to 1 indicates a very strong similarity between the two vectors.


```python
v1.dot(dv)
```




    np.float32(0.32332397)



This score is pretty good, but not great. This makes sense:  the question asks about whether a student can join the course after the start date; the answer, while related to the concept of joining the course, does not specifically address the question of joining the course after the start date.  

Let's try another question:


```python
q2 = 'How to install Docker on Windows?'
v2 = model.encode(q2)
```


```python
v2.dot(dv)
```




    np.float32(0.019730523)



Here the score is much lower.  If we look at the original value for ```d``` ("You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."), we can see that ```d``` does not contain content related to installing Docker on Windows.  

## Embedding Our Dataset

The simple example above shows a trivial case where a question is compared to a single answer for similarity.  For our search engine to be useful, we need to be able to pull in a lot more information.  

In the previous module, we created and used an ingestion script to pull in the course FAQ information into our pipeline.  Let's pull the script now so that we can use it.

### Loading the data


```python
# !wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/ingest.py
```

### Generating embeddings


```python
from ingest import load_faq_data

documents = load_faq_data()
```

Let's see what we have, taking look at a single item:


```python
documents[10]
```




    {'course': 'data-engineering-zoomcamp',
     'section': 'General Course-Related Questions',
     'question': 'Course: How many hours per week am I expected to spend on this course?',
     'answer': 'It depends on your background and previous experience with modules. It is expected to require about 5 - 15 hours per week.\n\nYou can also calculate it yourself using [this data](https://github.com/DataTalksClub/zoomcamp-analytics/tree/main/data/de-zoomcamp-2023) and then update this answer.',
     'doc_id': '316180784f'}



This is a python dictionary containing a single question and answer from the FAQ database.  We need to turn this python dictionary into the proper text that we can embed into vector space.  


```python
texts = []

for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)
```

Taking a look at one element:


```python
texts[10]
```




    'Course: How many hours per week am I expected to spend on this course? It depends on your background and previous experience with modules. It is expected to require about 5 - 15 hours per week.\n\nYou can also calculate it yourself using [this data](https://github.com/DataTalksClub/zoomcamp-analytics/tree/main/data/de-zoomcamp-2023) and then update this answer.'




```python
# check the number of texts within texts

len(texts)
```




    1350



We have to do this for every individual text (Q&A item) in this in the FAQ corpus, which contains 1,350 texts.  If we try to load them all at once, it will take a long time and we can't see what's happening inside.  Ingesting the texts in batches allows us to watch how it's going as the process proceeds. 


```python
# Shows a progress bar so we can see how long it takes to encode all the texts

from tqdm.auto import tqdm
```

Let's chunk the dataset into batches of 50 and encode each batch:


```python
batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

len(vectors)
```


      0%|          | 0/27 [00:00<?, ?it/s]





    1350




```python
scores = []

for i in range(len(vectors)):
    score = v1.dot(vectors[i])
    scores.append(score)
```


```python
import numpy as np
X = np.array(vectors)
```


```python
X
```




    array([[-0.02670618, -0.12245759,  0.01594416, ..., -0.00230645,
            -0.11218396, -0.02365561],
           [-0.01099554, -0.1107475 , -0.02536939, ...,  0.09022234,
            -0.02697358,  0.01975662],
           [-0.08896555, -0.06128182,  0.00775604, ...,  0.04059714,
             0.00479282, -0.02745941],
           ...,
           [-0.03652922,  0.01415427, -0.06838643, ...,  0.04316792,
             0.08105534, -0.02148628],
           [-0.13091592, -0.06990599, -0.00931885, ..., -0.00044336,
            -0.01285727,  0.01426919],
           [-0.07984785,  0.0192698 ,  0.0254498 , ..., -0.03368026,
            -0.01884023,  0.05837052]], shape=(1350, 384), dtype=float32)




```python
scores = X.dot(v1)
```




```python
idx = np.argmax(scores)
idx, scores[idx]
```




    (np.int64(2), np.float32(0.7629411))






```python

documents[553]
```




    {'course': 'llm-zoomcamp',
     'section': 'Module 1: RAG',
     'question': 'OpenAI: Error when running OpenAI responses.create command',
     'answer': 'You may receive the following error when running the OpenAI `responses.create` command due to insufficient credits in your OpenAI account:\n\n```\nOpenAI API Error: Insufficient credits\n```',
     'doc_id': 'f5df151c59'}






```python
top5 = np.argsort(scores)[-5:]
top5 = top5[::-1]
```


```python

scores[top5]
```




    array([0.7629411 , 0.7579371 , 0.71921337, 0.6536313 , 0.5601001 ],
          dtype=float32)




```python
for idx in top5:
    print(scores[idx])
    print(documents[idx])
    print()
```

    0.7629411
    {'course': 'data-engineering-zoomcamp', 'section': 'General Course-Related Questions', 'question': 'Course: Can I still join the course after the start date?', 'answer': "Yes, even if you don't register, you're still eligible to submit the homework.\n\nBe aware, however, that there will be deadlines for turning in homeworks and the final projects. So don't leave everything for the last minute.", 'doc_id': '3f1424af17'}
    
    0.7579371
    {'course': 'mlops-zoomcamp', 'section': 'General Course-Related Questions', 'question': 'Course - Can I still join the course after the start date?', 'answer': "Yes, even if you don't register, you're still eligible to submit the homeworks as long as the form is still open and accepting submissions.\n\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything to the last minute.", 'doc_id': '2d8b16c2a0'}
    
    0.71921337
    {'course': 'machine-learning-zoomcamp', 'section': 'General Course-Related Questions', 'question': 'The course has already started. Can I still join it?', 'answer': 'Yes, you can. Even though you missed the start date, you can register for the course. You won’t be able to submit some of the homeworks, but you can still take part in the course.\n\nIn order to get a certificate, you need to submit 2 out of 3 course projects and review 3 peers by the deadline. It means that if you join the course at the end of November and manage to work on two projects, you will still be eligible for a certificate.', 'doc_id': '41aabbd7c5'}
    
    0.6536313
    {'course': 'llm-zoomcamp', 'section': 'General Course-Related Questions', 'question': 'I just discovered the course. Can I still join?', 'answer': 'Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions.', 'doc_id': '74eb249bbf'}
    
    0.5601001
    {'course': 'data-engineering-zoomcamp', 'section': 'General Course-Related Questions', 'question': 'Course - Can I follow the course after it finishes?', 'answer': 'Yes, we will keep all the materials available, so you can follow the course at your own pace after it finishes.\n\nYou can also continue reviewing the homeworks and prepare for the next cohort. You can also start working on your final capstone project.', 'doc_id': '068529125b'}
    



```python
top5
```




    array([  2, 625, 907, 538,   7])




```python
top5 = np.argsort(-scores)[:5]
```


```python
top5
```




    array([  2, 625, 907, 538,   7])




```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=['course'])
vindex.fit(X, documents)
```




    <minsearch.vector.VectorSearch at 0x16a0a9d30>




```python
vindex.search(v1, num_results=5, filter_dict={'course': 'llm-zoomcamp'})
```




    [{'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'I just discovered the course. Can I still join?',
      'answer': 'Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions.',
      'doc_id': '74eb249bbf'},
     {'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'When will the course be offered next?',
      'answer': 'Summer 2027.',
      'doc_id': 'bd31146b0e'},
     {'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Certificate: Can I follow the course in a self-paced mode and get a certificate?',
      'answer': 'No, you can only get a certificate if you finish the course with a "live" cohort.\n\nWe don\'t award certificates for the self-paced mode. The reason is you need to peer-review 3 capstone(s) after submitting your project.\n\nYou can only peer-review projects at the time the course is running; after the form is closed and the peer-review list is compiled.',
      'doc_id': '69d122f12e'},
     {'course': 'llm-zoomcamp',
      'section': 'Module 1: RAG',
      'question': 'Can I run the course locally instead of Codespaces?',
      'answer': 'Yes. Codespaces is just the easiest way for everyone to start with the same environment.\n\nYou can run the course locally if you are comfortable setting up Python, `uv`, Jupyter, Docker, and any other tools needed for the module.\n\nIf you run locally, make sure you document your setup and keep your environment reproducible.',
      'doc_id': 'aa310de435'},
     {'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?',
      'answer': "You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.",
      'doc_id': '977bf7786c'}]



## RAG with Vector Search
_(The text below is from the lesson, with minor edits.)_

In module 1, we built a RAG pipeline with three steps:

    def rag(question):
        search_results = search(question)
        user_prompt = build_prompt(question, search_results)
        return llm(user_prompt)

The search step used keyword search. Now we swap in vector search. Because RAG is modular, search is the only step we touch. The build prompt and the LLM call stay exactly as before.

In module 1 we put all the RAG logic into a ```RAGBase``` helper class. It has ```search```, ```build_prompt```, and ```llm``` methods, so we only need to override ```search```.

Download ```rag_helper.py``` (and ```ingest.py``` if you didn't get it earlier) into your project:


```python
# !wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/rag_helper.py
```


```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()
```


```python
from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)
```


```python
from rag_helper import RAGBase

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
)
```


```python
query = 'I just found out about the program, can I still sign up?'
assistant.rag(query)
```




    'Yes, but if you want to receive a certificate, you need to submit your project while submissions are still being accepted.'




```python
class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {'course': self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )
```


```python
vector_assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=openai_client,
)
```


```python
query = 'I just found out about the program, can I still sign up?'
vector_assistant.rag(query)
```




    'Yes, but if you want to receive a certificate, you need to submit your project while submissions are still being accepted.'



## Vector Search with sqlitesearch 
_(The text below draws on the lesson text but includes my notes from the video lessons as well.)_

In the previous section we used ```minsearch``` for vector search.

It works, but it has three problems:

1. It rebuilds the index on every startup
2. It keeps everything in memory
3. It searches by brute force

**Rebuilding the index every time**--With text search, indexing was fast because we didn't embed anything. With vector search, indexing runs a neural network over every document, so it takes a bit of time our dataset. 

**Keeping everything in memory**--Keeping everything in memory is fine here because our dataset is relative small, but a larger dataset would require too much space.

**Brute-force search**--For every query we compare the query vector against every single document. With 1,000 documents, this is fine; in fact, it's probably faster than anything 'smarter'. But as the dataset grows past 10,000 or so, the process slows down.  This is *exact nearest neighbor (NN) search*. We score the query against every document and pick the top ones. It always finds the true top matches, but it pays for that by touching everything.  

*Approximate nearest neighbor (ANN) search* takes a shortcut. Instead of comparing against everything, it first narrows down to a region of likely matches. Then it scores only within that region. It may miss the absolute best match, but the results are still good and it's much faster.  That's what we will implement below.  

### What is sqlitesearch and why are we using it?

We'll be using ```sqlitesearch```, a vector search library Alexey created.  It has sqlite under the hood, so it's a proper database.  Also, everything is persisted--meaning that you can put vectors in one process and load vectors from another process.    We split the process into two parts:  ingestion and deployment.  The FAQ index is created once in the ingestion process and is ready to go every time we start up another search. 


```python
from sqlitesearch import VectorSearchIndex

vs_index = VectorSearchIndex(
    keyword_fields=['course'],
    mode='ivf',
    db_path='faq_vectors2.db'
)
```


```python
vs_index.clear()
```




    <sqlitesearch.vector.index.VectorSearchIndex at 0x306c342f0>




```python
vs_index.fit(vectors, documents)
```




    <sqlitesearch.vector.index.VectorSearchIndex at 0x306c342f0>




```python
query = 'I just discovered the course. Can I still join it?'
query_vector = model.encode(query)

results = vs_index.search(query_vector, num_results=5)
```


```python
results = vs_index.search(
    query_vector,
    filter_dict={'course': 'llm-zoomcamp'},
    num_results=5
)
```


```python
results
```




    [{'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'I just discovered the course. Can I still join?',
      'answer': 'Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions.',
      'doc_id': '74eb249bbf'},
     {'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Certificate: Can I follow the course in a self-paced mode and get a certificate?',
      'answer': 'No, you can only get a certificate if you finish the course with a "live" cohort.\n\nWe don\'t award certificates for the self-paced mode. The reason is you need to peer-review 3 capstone(s) after submitting your project.\n\nYou can only peer-review projects at the time the course is running; after the form is closed and the peer-review list is compiled.',
      'doc_id': '69d122f12e'},
     {'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'When will the course be offered next?',
      'answer': 'Summer 2027.',
      'doc_id': 'bd31146b0e'},
     {'course': 'llm-zoomcamp',
      'section': 'Module 1: RAG',
      'question': 'Can I run the course locally instead of Codespaces?',
      'answer': 'Yes. Codespaces is just the easiest way for everyone to start with the same environment.\n\nYou can run the course locally if you are comfortable setting up Python, `uv`, Jupyter, Docker, and any other tools needed for the module.\n\nIf you run locally, make sure you document your setup and keep your environment reproducible.',
      'doc_id': 'aa310de435'},
     {'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?',
      'answer': "You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.",
      'doc_id': '977bf7786c'}]




```python
vs_index.close()
```

### Implementing the vector search process using persistent index



```python
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

model = SentenceTransformer('all-MiniLM-L6-v2')

vs_index = VectorSearchIndex(
    keyword_fields=['course'],
    mode='ivf',
    db_path='faq_vectors2.db'
)
```

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]



```python
vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```


```python
query_vector = model.encode("How do I run Kafka?")
results = vs_index.search(query_vector, num_results=5)
```

Note that while we did not need to do the ingestion step, we still had to do the encoding (vector embedding).


```python
results
```




    [{'course': 'data-engineering-zoomcamp',
      'section': 'Module 7: Streaming',
      'question': 'Java Kafka: How to run producer/consumer/kstreams/etc in terminal',
      'answer': 'In the project directory, run:\n\n```bash\njava -cp build/libs/<jar_name>-1.0-SNAPSHOT.jar:out src/main/java/org/example/JsonProducer.java\n```',
      'doc_id': '5ca6890c1a'},
     {'course': 'data-engineering-zoomcamp',
      'section': 'Module 7: Streaming',
      'question': 'Java Kafka: When running the producer/consumer/etc java scripts, no results retrieved or no message sent',
      'answer': 'For example, when running `JsonConsumer.java`, you might see:\n\n```\nConsuming form kafka started\n\nRESULTS:::0\n\nRESULTS:::0\n\nRESULTS:::0\n```\n\nOr when running `JsonProducer.java`, you might encounter:\n\n```\nException in thread "main" java.util.concurrent.ExecutionException: org.apache.kafka.common.errors.SaslAuthenticationException: Authentication failed\n```\n\n**Solution:**\n\n1. Ensure the `StreamsConfig.BOOTSTRAP_SERVERS_CONFIG` in the scripts located at `src/main/java/org/example/` (e.g., `JsonConsumer.java`, `JsonProducer.java`) is pointing to the correct server URL (e.g., `europe-west3` vs `europe-west2`).\n\n2. Verify that the cluster key and secrets are updated in `src/main/java/org/example/Secrets.java` (`KAFKA_CLUSTER_KEY` and `KAFKA_CLUSTER_SECRET`).',
      'doc_id': 'cd8a62fc55'},
     {'course': 'data-engineering-zoomcamp',
      'section': 'Module 7: Streaming',
      'question': 'Confluent Kafka: Where can I find schema registry URL?',
      'answer': 'In [Confluent Cloud](https://confluent.cloud/):\n\n- Navigate to your Environment (e.g., default or a custom name).\n- Use the right navigation bar to find "Stream Governance API."\n- The URL can be found under "Endpoint."\n- Create credentials from the Credentials section below it.',
      'doc_id': '30fbb4f5b8'},
     {'course': 'data-engineering-zoomcamp',
      'section': 'Module 7: Streaming',
      'question': 'Python Kafka: ./spark-submit.sh streaming.py Error: py4j.protocol.Py4JJavaError: An error occurred while calling None.org.apache.spark.api.java.JavaSparkContext.',
      'answer': "Make sure your Java version is 11 or 8.\n\n- Check your version by:\n\n  ```bash\n  java --version\n  ```\n\n- Check all your installed Java versions by:\n\n  ```bash\n  /usr/libexec/java_home -V\n  ```\n\n- If you already have Java 11 but it's not set as the default, select the specific version by:\n\n  ```bash\n  export JAVA_HOME=$(/usr/libexec/java_home -v 11.0.22)\n  ```\n\n  (or another version of 11)",
      'doc_id': '0e8bce921a'},
     {'course': 'data-engineering-zoomcamp',
      'section': 'Module 7: Streaming',
      'question': 'Java Kafka: <project_name>-1.0-SNAPSHOT.jar errors: package xxx does not exist even after gradle build',
      'answer': 'In my setup, all of the dependencies listed in `build.gradle` were not installed in `<project_name>-1.0-SNAPSHOT.jar`.\n\nSolution:\n\n1. In the `build.gradle` file, add the following at the end:\n   \n   ```groovy\n   shadowJar {\n       archiveBaseName = "java-kafka-rides"\n       archiveClassifier = \'\'\n   }\n   ```\n\n2. In the command line, run:\n   \n   ```bash\n   gradle shadowjar\n   ```\n\n3. Execute the script from `java-kafka-rides-1.0-SNAPSHOT.jar` created by the shadowjar.',
      'doc_id': 'd01cbfa9cb'}]




```python
# filter by course

results = vs_index.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```


```python
results
```




    [{'course': 'llm-zoomcamp',
      'section': 'Capstone Project',
      'question': 'Project: what does "reproducibility" mean — do reviewers need access to my API keys?',
      'answer': "Never share API keys or hosted-service credentials in your repo. Reproducibility means a peer reviewer can clone the repo and follow your README to recreate the system from scratch — using their own credentials.\n\nConcretely:\n\n- Provide a script (or notebook) that ingests the dataset and (re)builds the search index locally.\n- Ship a `.env.example` with the variable names but no values; have the reviewer create their own `.env` with their own keys. Keep `.env` in `.gitignore`.\n- Use a cheap model (`gpt-4o-mini`, Groq, etc.) so reviewers don't burn through credits when running your project.\n- Pin dependency versions (`requirements.txt` / `pyproject.toml` lock file) and document the Python version (and Docker version, if used).",
      'doc_id': 'e5d8a2c761'},
     {'course': 'llm-zoomcamp',
      'section': 'Module 1: Agentic RAG',
      'question': 'Any free models with tool use support?',
      'answer': "Several Groq models offer tool use, such as Deepseek R1 or Llama 4, all of which can be used for free for development.\n\nOther providers also support tool or function calling, including Mistral, Gemini, and some local Ollama models.\n\nYou'll typically need to adapt the code when not using OpenAI, because tool schemas and response shapes differ between providers.\n\nFor more details, see the [Groq Tool Use Documentation](https://console.groq.com/docs/tool-use).",
      'doc_id': '0d74a3616f'},
     {'course': 'llm-zoomcamp',
      'section': 'Capstone Project',
      'question': 'How can I find some good ideas or datasets for the project?',
      'answer': 'Please check [this GitHub page](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md) for several ideas and datasets that could be used for the project, along with tips and guidelines.',
      'doc_id': 'e76a70cde3'},
     {'course': 'llm-zoomcamp',
      'section': 'Module 1: RAG',
      'question': 'WSL2: ResponseError: model requires more system memory (X.X GiB) than is available (Y.Y GiB). My system has more than X.X GiB.',
      'answer': 'Your WSL2 is set to use Y.Y GiB, not all your computer memory. To allocate more RAM, follow these steps:\n\n1. Create a `.wslconfig` file under your Windows user profile directory: `C:\\Users\\YourUsername\\.wslconfig`.\n\n2. Include the desired RAM allocation in the file:\n\n   ```ini\n   [wsl2]\n   memory=8GB\n   ```\n\n3. Restart WSL using the command:\n\n   ```bash\n   wsl --shutdown\n   ```\n\n4. Run the `free` command in WSL to verify the changes.\n\nFor more details, read [this article](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).',
      'doc_id': 'f3dd94f323'},
     {'course': 'llm-zoomcamp',
      'section': 'Module 1: RAG',
      'question': 'OpenAI: How much will I have to spend to use the Open AI API?',
      'answer': 'Using the OpenAI API for the course should cost very little. You can recharge starting from $5, but initial usage is usually fractions of one cent.',
      'doc_id': '554d0eb78b'}]



### Using sqlitesearch vector search in RAG
Let's use our persistent vector index in RAG.


```python
from rag_helper import RAGBase
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )

vector_assistant = RAGVector(
    embedder=model,
    index=vs_index,
    llm_client=openai_client,
)
```


```python
vector_assistant.rag("the program has already begun, can I still sign up?")
```




    "Yes, you can still join. You don’t need to wait for a confirmation email—you're accepted, and you can start learning and submitting homework while the form is open."




```python
vs_index.close()
```

### Comparing minsearch and sqlitesearch for vector search

Here is how the two approaches compare:

* minsearch ```VectorSearch```: in-memory (numpy), exact cosine similarity, must re-compute embeddings on startup, good for experiments and notebooks
* sqlitesearch ```VectorSearchIndex```: persistent (SQLite ```.db``` file), ANN (LSH/IVF/HNSW) with exact rerank, can open an existing index, good for projects and persistence

Regarding ```sqlitesearch```, this is not something you would use for commercial deployment, since it was created for educational purposes.  However, for coursework, it is useful:  SQLite is a lightweight database and many hosts have a free SQLite database option that may work well for our purposes.


## Vector Search with PGVector
_(Text is from the lesson)_

Many real databases can do vector search. Elasticsearch has it, and there are dedicated stores like Qdrant and Chroma. We'll go with Postgres. Most of us already run it at work, and the data engineering course uses it too. The concept is the same as with sqlitesearch; only the database under the hood changes.

pgvector is the PostgreSQL extension that makes this work. Install it and Postgres can do vector similarity search. On top of that you get the usual production features, like concurrent access, transactions, and large datasets.

We'll run Postgres with pgvector in Docker.

### Starting Postgres with pgvector
We will run Postgres in Docker for this part of the lesson.  First we need to spin up a Docker container with the relevant extensions/commands:  

Run in terminal:

    docker run -it \
        --name pgvector \
        -e POSTGRES_USER=user \
        -e POSTGRES_PASSWORD=pswd \
        -e POSTGRES_DB=faq \
        -v pgvector_data:/var/lib/postgresql/data \
        -p 5432:5432 \
        pgvector/pgvector:pg17


This image has the pgvector extension pre-installed. The -v flag creates a named volume so data persists across container restarts.

Installing the Python client
Install the driver:

    uv add psycopg

We'll use ```psycopg``` (v3) to connect and run queries. _(Thank goodness, because ```psycopg2``` was a pain to install last time I was using Docker with PostgreSQL.)_ psycopg (v3) is different from psycopg2 - psycopg v3 supports ```conn.execute()``` directly without creating a cursor.

### Preparing the data
We need the FAQ documents and their embeddings.

Here's what we did in previous units as one script:


```python
from tqdm.auto import tqdm

from ingest import load_faq_data
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_faq_data()
texts = [doc["question"] + " " + doc["answer"] for doc in documents]

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)
```

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]



      0%|          | 0/27 [00:00<?, ?it/s]


Now we connect to Postgres:


```python
import psycopg

conn = psycopg.connect(
    "postgresql://user:pswd@localhost:5432/faq"
)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
```




    <psycopg.Cursor [COMMAND_OK] [INTRANS] (host=localhost user=user database=faq) at 0x136a72690>



The second line activates pgvector. The Docker image we started isn't plain Postgres; it ships the extension inside, and this turns it on. It adds the vector column type and the similarity search operators.

### Creating a table
Create a table for storing documents with their embeddings using SQL:


```python
conn.execute("""
    DROP TABLE IF EXISTS documents
""")

conn.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        course TEXT,
        section TEXT,
        question TEXT,
        answer TEXT,
        embedding vector(384)
    )
""")
```




    <psycopg.Cursor [COMMAND_OK] [INTRANS] (host=localhost user=user database=faq) at 0x10390bd10>



The ```vector```(384) column stores our 384-dimensional embeddings from ```all-MiniLM-L6-v2```.

### Inserting documents with embeddings
Let's insert the documents and their vectors into PGVector:


```python
# Function to convert a vector to a string format suitable for PostgreSQL vector type (used in command that follows)
def vec_to_str(vector): 
    return "[" + ",".join(str(x) for x in vector) + "]" 

# Iterate over each document and its corresponding vector and insert them into the PostgreSQL table
for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):  
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector) 
        """,
        (doc["course"], doc["section"], doc["question"], doc["answer"],
         vec_to_str(vec))
    )

conn.commit()
```


      0%|          | 0/1350 [00:00<?, ?it/s]


We loop over the documents and insert each one with its embedding. We hand Postgres the vector as text, so the ```::vector``` cast tells it to parse that string back into a vector. We call ```conn.commit()``` to persist the changes.

### Searching with cosine similarity
Search with a query:


```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)
query_str = vec_to_str(query_vector)
```

Search for the most similar documents:


```python
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, query_str)
).fetchall()

for row in results:
    print(f"[{row[0]}] {row[1]} (similarity: {row[3]:.4f})")
```

    [llm-zoomcamp] I just discovered the course. Can I still join? (similarity: 0.8365)
    [machine-learning-zoomcamp] The course has already started. Can I still join it? (similarity: 0.6904)
    [mlops-zoomcamp] Course - Can I still join the course after the start date? (similarity: 0.6043)
    [data-engineering-zoomcamp] Course: Can I still join the course after the start date? (similarity: 0.5959)
    [data-engineering-zoomcamp] Course: Can I get support if I take the course in the self-paced mode? (similarity: 0.5927)


The ```<=>``` operator computes cosine distance (1 - cosine similarity). We order by ascending distance, so the closest vectors come first.

### Filtering by course
Because this is plain SQL, filtering by course is one extra ```WHERE``` clause:


```python
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE course = %s
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, "llm-zoomcamp", query_str)
).fetchall()
```

### Creating an index for faster search
So far this runs brute-force search, comparing our query against every row. For our small dataset that's fine.

For a larger one, create an HNSW index to switch to approximate search:


```python
conn.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
""")
```




    <psycopg.Cursor [COMMAND_OK] [INTRANS] (host=localhost user=user database=faq) at 0x1320e6510>



This builds an HNSW (Hierarchical Navigable Small World) index, the same state-of-the-art algorithm dedicated vector databases use. It makes search faster, at the cost of a small accuracy trade-off.

### Wrapping it in a function
Let's wrap the search logic in a reusable function:


```python
def pgvector_search(query, course="llm-zoomcamp", num_results=5):
    query_vector = model.encode(query)
    query_str = vec_to_str(query_vector)
    rows = conn.execute(
        """
        SELECT course, section, question, answer
        FROM documents
        WHERE course = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (course, query_str, num_results)
    ).fetchall()

    return [
        {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
        for r in rows
    ]
```


```python
results = pgvector_search("How do I join the course?")
```


```python
results
```




    [{'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'I just discovered the course. Can I still join?',
      'answer': 'Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions.'},
     {'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'When will the course be offered next?',
      'answer': 'Summer 2027.'},
     {'course': 'llm-zoomcamp',
      'section': 'Module 1: RAG',
      'question': 'Can I run the course locally instead of Codespaces?',
      'answer': 'Yes. Codespaces is just the easiest way for everyone to start with the same environment.\n\nYou can run the course locally if you are comfortable setting up Python, `uv`, Jupyter, Docker, and any other tools needed for the module.\n\nIf you run locally, make sure you document your setup and keep your environment reproducible.'},
     {'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Certificate: Can I follow the course in a self-paced mode and get a certificate?',
      'answer': 'No, you can only get a certificate if you finish the course with a "live" cohort.\n\nWe don\'t award certificates for the self-paced mode. The reason is you need to peer-review 3 capstone(s) after submitting your project.\n\nYou can only peer-review projects at the time the course is running; after the form is closed and the peer-review list is compiled.'},
     {'course': 'llm-zoomcamp',
      'section': 'Module 1: RAG',
      'question': 'OpenAI: Do I have to subscribe and pay for Open AI API for this course?',
      'answer': "No, you don't have to pay for this service in order to complete the course homeworks. You can use free or low-cost alternatives listed in the course GitHub repo.\n\nSee the course list of [OpenAI API alternatives](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/awesome-llms.md#openai-api-alternatives)."}]



### Using it in RAG
We take the same ```search``` function from above and move it into a class. We pass the Postgres connection instead of an index. We set ```index=None``` because ```RAGBase``` expects an index and would complain otherwise.

The class overrides ```search``` to query PGVector:


```python
from rag_helper import RAGBase

class RAGPgVector(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]
```

Initialize OpenAI client:


```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()
```

Create the vector assistant:


```python
vector_assistant = RAGPgVector(
    embedder=model,
    conn=conn,
    llm_client=openai_client,
)
```


```python
vector_assistant.rag("the program has already begun, can I still sign up?")
```




    'Yes, you can still join. You don’t need a confirmation email, and you can start learning and submitting homework while the form is open. If you want a certificate, make sure you submit your project while submissions are still being accepted.'



### Using PGVector
Here's how PGVector compares with the two tools we used earlier:

* ```minsearch```: no setup, in-memory, best for notebooks and experiments
* ```sqlitesearch```: no setup, SQLite file persistence, best for pet projects
* ```PGVector```: requires Docker, Postgres database with concurrent access, handles millions of records, best for production systems

Reach for PGVector when you need production features:

* concurrent reads and writes
* transactions
* integration with an existing Postgres-based application

## Using ONNX Runtime instead of PyTorch

_(from the lesson text)_

When you move to production, you want to cut overhead, both the dependencies and the size of your deployment. sentence-transformers drags in PyTorch plus a pile of Nvidia libraries, which is a lot. ONNX Runtime serves the same model without that weight.

To put a number on it, I created two empty projects. In one I ran uv add sentence-transformers, in the other I set up ONNX Runtime.

Then I measured the virtual environment sizes:

* sentence-transformers: 4.8 GB, 58 packages
* ONNX Runtime: 147 MB, 27 packages

That's 33x smaller for the same embeddings and the same results. Often we don't even convert the model ourselves. Someone has usually published an ONNX version we can download.

For development and experiments, sentence-transformers is fine. For production you want the lighter option.

Let's create a separate project for this lesson (run in terminal):

    mkdir llm-zoomcamp-onnx && cd llm-zoomcamp-onnx
    uv init --no-workspace
    uv add onnxruntime tokenizers numpy tqdm minsearch
    uv add --dev huggingface-hub jupyter

```huggingface-hub``` is only needed to download the model. At runtime we'll need ```onnxruntime```, ```tokenizers```, and ```numpy```.

Then register a kernel for this project (run in terminal):

    uv run python -m ipykernel install --user --name llm-zoomcamp-onnx --display-name "llm-zoomcamp-onnx"

### Downloading the model
We'll use the download.py script from the embed/ directory to fetch the ONNX model from HuggingFace.

Copy it to your project, then run:
```uv run python download.py```

This creates:
  models/
    Xenova/
      all-MiniLM-L6-v2/
        tokenizer.json
        model.onnx

You only run this once. After that, the model files are local.

Add the models directory to .gitignore:

```models/```

### The Embedder class
We'll use the embedder.py script from the ```embed/``` directory for generating embeddings.

Copy it to your project as well.

Under the hood, it does four things:

1. Tokenize - convert text into integer IDs and attention masks
2. Run ONNX model - execute the model graph on CPU
3. Mean pooling - average the token embeddings, weighted by the attention mask
4. Normalize - divide by L2 norm so vectors can be compared with dot product

You don't need to follow every step inside ```embedder.py```. It gives us the same ```encode``` interface as before, with none of the PyTorch weight.



### Same pipeline, no PyTorch
Let's repeat the examples from earlier and confirm the numbers match.

First, comparing two queries against a document:


```python
from embedder import Embedder

embed = Embedder()

q1 = "Can I still join the course after the start date?"
q2 = "How to install Docker on Windows?"
d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."

v1 = embed.encode(q1)
v2 = embed.encode(q2)
dv = embed.encode(d)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[17], line 1
    ----> 1 from embedder import Embedder
          2 
          3 embed = Embedder()
          4 


    ModuleNotFoundError: No module named 'embedder'


## Vector Search with sqlitesearch 
(text below is verbatim from the lesson)

In the previous section we used minsearch for vector search.

It works, but it has three problems:

1. It rebuilds the index on every startup
2. It keeps everything in memory
3. It searches by brute force

With text search we never felt these. Indexing was fast because we didn't embed anything. With vector search, indexing runs a neural network over every document, so it takes a minute on our dataset. Keeping everything in memory is fine here, but a larger dataset would need too much space.

The third problem is brute-force search. For every query we compare the query vector against every single document. With 1,000 documents this is fine, probably even faster than anything smarter. But as the dataset grows past 10,000 or so, it gets slow, and we'll want an approximate method instead.

What we've done so far is exact nearest neighbor (NN) search. We score the query against every document and pick the top ones. It always finds the true top matches, but it pays for that by touching everything.

Approximate nearest neighbor (ANN) search takes a shortcut. Instead of comparing against everything, it first narrows down to a region of likely matches. Then it scores only within that region. It may miss the absolute best match, but the results are still good and it's much faster.
