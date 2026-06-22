# Module 1:  Agentic RAG 

## Table of Contents


- [Part 1: Building the foundation for our RAG agent](#part-1--building-the-foundation-for-our-rag-agent)
  - [What is RAG and why do we need it?](#what-is-rag-and-why-do-we-need-it)
  - [The Course FAQ Dataset](#the-course-faq-dataset)
  - [Search](#search)
  - [Building the Prompt](#building-the-prompt)
  - [The LLM](#the-llm)
  - [RAG Helper](#rag-helper)
  - [Persistent RAG](#persistent-rag)
  - [Demonstrating Persistent RAG in Action: Exercise](#demonstrating-persistent-rag-in-action--exercise)
- [Part 2: Agents](#part-2--agents)
  - [Quick RAG revision](#quick-rag-revision)
  - [The agentic approach](#the-agentic-approach)
  - [Defining the tools](#defining-the-tools)
  - [The Agentic Loop](#the-agentic-loop)
  - [Encouraging multiple searches](#encouraging-multiple-searches)
  - [ToyAIKit](#toyaikit)
  - [Other Frameworks](#other-frameworks)
  - [A note about avoiding agents when a simpler tool will do the job](#a-note-about-avoiding-agents-when-a-simpler-tool-will-do-the-job)


## Part 1:  Building the foundation for our RAG agent 


In Module 1 of the course, we learn what LLMs are and build a simple RAG pipeline using keyword search.  Then we make it agentic, so the LLM decides when and what to search instead of running a fixed pipeline.  

This notebook implements the first part of this workflow.

Reference:  https://github.com/DataTalksClub/llm-zoomcamp/tree/main/01-agentic-rag


```python
from dotenv import load_dotenv
load_dotenv()
```




    True




```python
from openai import OpenAI
openai_client = OpenAI()
```


### What is RAG and why do we need it?


#### Plain LLMs lack our data
First, let's define a function to talk to the LLM:


```python
def llm(prompt):
    response = openai_client.responses.create(
        model='gpt-5.4-mini',
        input=prompt
    )
    return response.output_text
```

This is our black box - text goes in, text comes out.

Let's test it:


```python
llm("hey what's up?")
```




    'Hey! Not much—just here and ready to help. What’s going on?'



It replies with something. The LLM works.

Ask it a course-specific question:


```python
question = "I just discovered the course.  Can I join now?"
answer = llm(question)
print(answer)
```

    Maybe — it depends on the course’s enrollment policy and whether it’s still open.
    
    A good next step is to:
    1. Check the course page for the enrollment deadline or “late enrollment” info.
    2. Contact the instructor or course support and ask if you can still join.
    3. If there’s a waitlist, ask whether spots are still available.
    
    If you want, I can help you draft a short message asking to join late.


The LLM gives a generic answer. It might say "you can usually join" or "check the course website." It doesn't know about our specific Zoomcamp courses, their enrollment policies, or their schedules. It tries to be helpful, but has no idea about actual enrollment status or policies.

This is different from a question like "how do I cook salmon?" - the LLM knows the answer because cooking salmon is common knowledge. But our courses are not in the training data.

#### Adding context manually

More context can fix this. The FAQ website has questions and answers about our courses.

Copy some of that content into the prompt:


```python
context = """
I just discovered the course. Can I still join?
Yes, but if you want to receive a certificate, you need to submit your project while we're still accepting submissions.

Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?
You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.

What is the video/zoom link to the stream for the "Office Hours" or live/workshop sessions?
The zoom link is only published to instructors/presenters/TAs. Students participate via YouTube Live and submit questions to Slido.

Cloud alternatives with GPU
Check the quota and reset cycle carefully. Potential options include Google Colab, Kaggle, Databricks.
"""
```

Build a prompt that includes both the question and the context:



```python
prompt = f"""
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."

Question:
{question}

Context:
{context}
"""
```

Instead of sending the raw question to the LLM, we send this prompt:


```python
print(prompt)
```

    
    Your task is to answer questions from the course participants
    based on the provided context.
    
    Use the context to find relevant information and provide accurate
    answers. If the answer is not found in the context,
    respond with "I don't know."
    
    Question:
    I just discovered the course.  Can I join now?
    
    Context:
    
    I just discovered the course. Can I still join?
    Yes, but if you want to receive a certificate, you need to submit your project while we're still accepting submissions.
    
    Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?
    You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.
    
    What is the video/zoom link to the stream for the "Office Hours" or live/workshop sessions?
    The zoom link is only published to instructors/presenters/TAs. Students participate via YouTube Live and submit questions to Slido.
    
    Cloud alternatives with GPU
    Check the quota and reset cycle carefully. Potential options include Google Colab, Kaggle, Databricks.
    
    


After that, the answer is correct: "Yes, you can still join. If you want to receive a certificate, you need to submit your project while submissions are still open."

This is the answer we actually want to give to our students. What we just did is nothing but RAG.


```python
question = 'I just discovered the course. Can I still join?'
answer = llm(prompt)
print(answer)
```

    Yes, you can still join. If you want to receive a certificate, make sure to submit your project while submissions are still being accepted.


#### Retrieval plus generation

RAG stands for Retrieval-Augmented Generation. Generation is the LLM producing text, and retrieval is search. We retrieve relevant documents from our knowledge base and use them to augment what the LLM generates. That search step is what gives the LLM the context it needs to answer correctly.

What we just did was naive. I knew in advance which FAQ entry held the answer and pasted it in by hand. What we want instead is to perform search automatically. We take the student's question, find the most relevant documents, and send those to the LLM.

In code, it looks like this:


```python
def rag(question):
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)
```

That's the entire architecture. It comes down to three components.

The pieces are search, the prompt, and the LLM:

* search
* prompt
* LLM



The LLM only sees the documents we hand it, so its answers are grounded in our data. If the right document is retrieved, the answer is good. If it's not, the LLM gets the wrong context and the answer is wrong. Your model is only as good as your retrieval, so search quality matters a lot for RAG.

The database and the LLM can be anything. In this course we use minsearch and then sqlitesearch for search, and OpenAI for the LLM. But you can swap any component for another and see what works better.

Because each piece is independent, RAG stays flexible. To use Anthropic instead of OpenAI, you swap the LLM call. To use Elasticsearch instead of minsearch, you swap the search call. Nothing else changes.

In the next section, we'll look at the dataset we'll use for our FAQ knowledge base.

[Back to TOC](#table-of-contents)

### The Course FAQ Dataset


Before we build the RAG pipeline, let's get familiar with the data we'll use as our knowledge base.

We run these courses every year, and students keep asking the same questions in Slack. We collected those into an FAQ so people can find answers before asking. Some courses have run for five cohorts, so the FAQ grows large and searching it by hand gets tedious. That's exactly the problem our RAG system will solve.

The FAQ data is available as JSON from the DataTalks.Club website. I maintain that site, so I made the data available at a JSON endpoint we can fetch directly.

Let's fetch it:


```python
import requests

docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()
```

This returns a list of courses. Each course has a path field that points to its FAQ data.

Let's fetch all the FAQ documents from all courses:


```python
documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"""{url_prefix}{course["path"]}"""

    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()

    documents.extend(course_data)

len(documents)
```




    1349



Each entry has:

* ```id``` - unique identifier
* ```course``` - course slug (e.g., machine-learning-zoomcamp)
* ```section``` - which section of the course
* ```question``` - the FAQ question
* ```answer``` - the FAQ answer

Let's look at one:


```python
documents[0]
```




    {'id': '9e508f2212',
     'course': 'data-engineering-zoomcamp',
     'section': 'General Course-Related Questions',
     'question': 'Course: When does the course start?',
     'answer': "A new cohort runs roughly January–April every year. For the current cohort's exact start date and registration link, check the [course repo README](https://github.com/DataTalksClub/data-engineering-zoomcamp).\n\n- Register via the link in the course repo before the cohort starts.\n- Join the [course Telegram channel](https://t.me/dezoomcamp) for announcements.\n- Join DataTalks.Club's [Slack](https://datatalks.club/docs/general/slack/) and the `#course-data-engineering` channel."}



Each course has a slug - a short identifier used in URLs. For example, machine-learning-zoomcamp, data-engineering-zoomcamp, etc. We'll use these slugs for filtering in search.

#### Using this data

In the RAG pipeline, this dataset is our knowledge base:

* We index all the documents (the search step)
* When a student asks a question, we search the index
* The search returns the most relevant FAQ entries
* We give those entries to the LLM as context
* The LLM generates an answer based on the context

The question and answer fields contain the text we'll search through. The course field lets us filter by course. For example, if a student asks about the data engineering course, we skip results from the ML course. The section field helps with ranking - knowing which part of the course a question belongs to is useful context.


#### A note on data preparation

In our case, the data is already prepared. Alexey maintains this FAQ website and made sure the data comes back in a convenient JSON format. So we don't need to do much to get it ready. He cleaned a lot of this data with the help of an LLM (a handy use case on its own).

In reality, data preparation is often the most time-consuming part of building a RAG system. It may be necessary to scrape websites, parse PDFs, and clean and chunk documents. Even though we don't see this part of the preparation process, Alexey spent a lot of time on these activities in advance.

We keep the focus on the GenAI side in this course. For our projects, we should expect to spend significant time on data preparation before we get to this point.

In the next section, we'll build the search index.

[Back to TOC](#table-of-contents)

### Search


#### Search basics

At its core, every search engine does the same thing. It takes a query, scores every document for similarity, and returns the top results.

For each document in the database, you compute this score. Then you rank all documents by score and return the top N. What makes a search engine different from another search engine is what sim actually computes.

* Text/lexical search (covered in this section): sim counts how many words the query and the document share. It looks at the surface form, the actual words, and matches them exactly.
* Vector/semantic search (covered in module 2): sim compares the meaning of the query and the document. Same function, different similarity measure.

Consider these two questions:

* "Can I still join the course after the start date?"
* "Is it possible to enroll late?"

They mean the same thing, but share almost no keywords. "Join" is not "enroll", "course" is absent, "start date" is not "late". A text search engine would struggle to match them, because it only sees words.

We'll see how vector search solves this later. For now, let's build text search with minsearch.

#### Indexing with minsearch

We already have the documents list from the previous section. Now let's index it.

Searching matters because we have around 1100 documents. Sending all of them to the LLM would be expensive and slow. The model would get confused with too much data. Search finds the most relevant documents to send instead.

There are many search libraries you can use - Apache Lucene, Elasticsearch, Solr, and others. But these are somewhat heavy. For example, to run Elasticsearch, you need to start a Docker container.

```minsearch``` is a simple in-memory search engine. It's lightweight, so it runs anywhere Python runs, including Google Colab where you can't start a Docker container. It's a toy implementation, not production ready, but it illustrates how search engines work and it gives good results.

The concepts in ```minsearch``` are the same as in Elasticsearch (which comes from Lucene): text fields, keyword fields, boosting, filtering. Alexei borrowed those terms from Elasticsearch on purpose, since he wanted a lightweight stand-in for it. So what we're learning here transfers directly.

We'll index the question, section, and answer fields as text (they'll be tokenized and ranked), and the course field as a keyword (for filtering).

The index tokenizes text fields and treats keyword fields as exact strings.

Text fields are the fields we search through. When we type a query, the search engine looks at these fields and tokenizes them. It splits text into words, lowercases them, removes stop words, and ranks the results by relevance. So question, section, and answer are text fields.

Keyword fields are for exact matching--similar to a SQL query like SELECT * FROM index WHERE course = 'data-engineering-zoomcamp'. The results must come from the specified course, no matter what ranking or boosting we do for text fields.

We use keyword fields to restrict the search space to a particular subset. In our case, we have four courses. Say someone is taking the LLM course and asks a question. They don't want answers from the MLOps or machine learning courses mixed in.


```python
from minsearch import Index

index = Index(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"]
)

index.fit(documents)
```




    <minsearch.minsearch.Index at 0x129ebbe00>



That's it; the index is built. The fit name comes from scikit-learn, where you fit a model on data. Here we fit an index on our documents.

#### Trying a search

Let's try a search with the question we used before:


```python
question = "I just discovered the course. Can I join now?"

search_results = index.search(
    question,
    boost_dict={"question": 2.0, "section": 0.5},
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)

search_results
```




    [{'id': '74eb249bbf',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'I just discovered the course. Can I still join?',
      'answer': 'Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions.'},
     {'id': '977bf7786c',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?',
      'answer': "You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date."},
     {'id': '69d122f12e',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Certificate: Can I follow the course in a self-paced mode and get a certificate?',
      'answer': 'No, you can only get a certificate if you finish the course with a "live" cohort.\n\nWe don\'t award certificates for the self-paced mode. The reason is you need to peer-review 3 capstone(s) after submitting your project.\n\nYou can only peer-review projects at the time the course is running; after the form is closed and the peer-review list is compiled.'},
     {'id': '04919992b3',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'How should I start the course and follow the weekly workflow?',
      'answer': 'Start with the [LLM Zoomcamp docs](https://datatalks.club/docs/courses/llm-zoomcamp/), the [general Zoomcamp logistics docs](https://datatalks.club/docs/courses/zoomcamp-logistics/), and the [LLM Zoomcamp GitHub repository](https://github.com/DataTalksClub/llm-zoomcamp).\n\nYou can start whenever you want. The videos and GitHub materials are available, and the deadlines are listed in the [course management platform](https://courses.datatalks.club/llm-zoomcamp-2026/).\n\nA typical workflow is:\n\n1. Watch the lesson videos.\n2. Work through the lesson notebooks/code.\n3. Read the homework instructions on GitHub.\n4. Submit answers through the course platform before the deadline.\n\nHomework is similar to the lesson flow, but uses a different dataset or slightly different task.'},
     {'id': 'bd31146b0e',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'When will the course be offered next?',
      'answer': 'Summer 2027.'}]



We get back 5 results from the LLM Zoomcamp FAQ. The best match is the FAQ entry "I just discovered the course. Can I still join?" which is exactly what we need.

Here are all the questions:


```python
[doc["question"] for doc in search_results]
```




    ['I just discovered the course. Can I still join?',
     'Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?',
     'Certificate: Can I follow the course in a self-paced mode and get a certificate?',
     'How should I start the course and follow the weekly workflow?',
     'When will the course be offered next?']



We see questions about joining the course, registration, certificates, and more. These are the candidate documents we'll send to the LLM.

We used boost_dict to give the question field more weight (2.0 instead of the default 1.0) and section less weight (0.5). Query words appearing in the question field is a stronger signal than them appearing in the section name.

We used filter_dict to only return results from the LLM Zoomcamp course. Without this filter, we'd get results from all four courses.


#### Boosting fields

Not all fields are equally important. The question field is usually more relevant than section for matching. Query words appearing in the question is a stronger signal than them appearing in the section name.

minsearch supports field boosting to reflect this:


```python
results = index.search(
    question,
    num_results=5,
    boost_dict={"question": 2.0, "section": 0.5}
)
```

All fields have a default boost of 1. Giving question a boost of 2 means it counts two times as much. Take a question about certificates. The word "certificate" in the question field now weighs twice what it does in the answer.

Giving section 0.5 means it counts half as much, since a match there tells us less. This is the same boosting mechanism used by Elasticsearch and Lucene.

#### Filtering by course

Sometimes you want to restrict the search to a specific course.

minsearch supports keyword filtering:


```python
results = index.search(
    question,
    num_results=5,
    filter_dict={"course": "mlops-zoomcamp"}
)
```

This only returns documents from the MLOps Zoomcamp. Try a few different queries and courses to get a feel for the results.


```python
[doc["question"] for doc in results]
```




    ['Course - Can I still join the course after the start date?',
     'Homework: Just found this course, can I still submit homeworks?',
     'I forgot if I registered, can I still join the zoomcamp?',
     'Certificate - Can I follow the course in a self-paced mode and get a certificate?',
     'Course: How do I start?']



#### Wrapping it in a function

Let's wrap the search in a search function - the first component of our RAG pipeline:


```python
def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5}
    filter_dict = {"course": course}

    return index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5
    )
```

By default it searches the LLM Zoomcamp FAQ.

You can pass a different course slug to search other courses:


```python
search_results = search(question)
```


```python
search_results
```




    [{'id': '74eb249bbf',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'I just discovered the course. Can I still join?',
      'answer': 'Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions.'},
     {'id': '977bf7786c',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?',
      'answer': "You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date."},
     {'id': '69d122f12e',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'Certificate: Can I follow the course in a self-paced mode and get a certificate?',
      'answer': 'No, you can only get a certificate if you finish the course with a "live" cohort.\n\nWe don\'t award certificates for the self-paced mode. The reason is you need to peer-review 3 capstone(s) after submitting your project.\n\nYou can only peer-review projects at the time the course is running; after the form is closed and the peer-review list is compiled.'},
     {'id': '04919992b3',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'How should I start the course and follow the weekly workflow?',
      'answer': 'Start with the [LLM Zoomcamp docs](https://datatalks.club/docs/courses/llm-zoomcamp/), the [general Zoomcamp logistics docs](https://datatalks.club/docs/courses/zoomcamp-logistics/), and the [LLM Zoomcamp GitHub repository](https://github.com/DataTalksClub/llm-zoomcamp).\n\nYou can start whenever you want. The videos and GitHub materials are available, and the deadlines are listed in the [course management platform](https://courses.datatalks.club/llm-zoomcamp-2026/).\n\nA typical workflow is:\n\n1. Watch the lesson videos.\n2. Work through the lesson notebooks/code.\n3. Read the homework instructions on GitHub.\n4. Submit answers through the course platform before the deadline.\n\nHomework is similar to the lesson flow, but uses a different dataset or slightly different task.'},
     {'id': 'bd31146b0e',
      'course': 'llm-zoomcamp',
      'section': 'General Course-Related Questions',
      'question': 'When will the course be offered next?',
      'answer': 'Summer 2027.'}]



[Back to TOC](#table-of-contents)

### Building the Prompt


The LLM doesn't see our documents unless we pass them in. So we need to build a prompt that includes the user's question and the search results.

When we build AI systems, we usually split the prompt into two parts:

* Instructions (also called the system prompt): this tells the LLM how to behave. It never changes, so it's the same for every request.
* User prompt: this changes with every request. It carries the actual question and the retrieved context.

We split them because the instructions are fixed and the user prompt is not. Keeping them apart makes the fixed part easy to reuse and the changing part easy to build fresh each time.

#### Instructions

The instructions tell the LLM its role and how to answer:


```python
INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""
```

This is what grounds the answer in our data and reduces hallucinations.

#### The user prompt template

The user prompt template has placeholders for the question and the context:


```python
USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""
```

#### Building the context

The context is a formatted string with all the search results:


```python
def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()
```

Each document becomes a block with the section, question, and answer. This format makes it easy for the LLM to read. We turned a list of dictionaries into one string. It's a small preprocessing step before we send the data to the LLM.

#### Building the prompt

Now we combine the question with the context into the user prompt:


```python
def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
    return prompt.strip()
```

Let's try it:


```python
prompt = build_prompt(question, search_results)

print(prompt)
```

    Question:
    I just discovered the course. Can I join now?
    
    Context:
    General Course-Related Questions
    Q: I just discovered the course. Can I still join?
    A: Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions.
    
    General Course-Related Questions
    Q: Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?
    A: You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.
    
    General Course-Related Questions
    Q: Certificate: Can I follow the course in a self-paced mode and get a certificate?
    A: No, you can only get a certificate if you finish the course with a "live" cohort.
    
    We don't award certificates for the self-paced mode. The reason is you need to peer-review 3 capstone(s) after submitting your project.
    
    You can only peer-review projects at the time the course is running; after the form is closed and the peer-review list is compiled.
    
    General Course-Related Questions
    Q: How should I start the course and follow the weekly workflow?
    A: Start with the [LLM Zoomcamp docs](https://datatalks.club/docs/courses/llm-zoomcamp/), the [general Zoomcamp logistics docs](https://datatalks.club/docs/courses/zoomcamp-logistics/), and the [LLM Zoomcamp GitHub repository](https://github.com/DataTalksClub/llm-zoomcamp).
    
    You can start whenever you want. The videos and GitHub materials are available, and the deadlines are listed in the [course management platform](https://courses.datatalks.club/llm-zoomcamp-2026/).
    
    A typical workflow is:
    
    1. Watch the lesson videos.
    2. Work through the lesson notebooks/code.
    3. Read the homework instructions on GitHub.
    4. Submit answers through the course platform before the deadline.
    
    Homework is similar to the lesson flow, but uses a different dataset or slightly different task.
    
    General Course-Related Questions
    Q: When will the course be offered next?
    A: Summer 2027.


The prompt is the bridge between search and the LLM. A bad prompt lets the LLM ignore the context and hallucinate. A good prompt keeps the answer grounded.

Prompt engineering is part art, part science. Experiment, try different things, and see what works. Later in the course, we will cover evaluation metrics to you can measure how well our prompt performs instead of guessing. For now, this template is a good starting point.

[Back to TOC](#table-of-contents)

### The LLM


The last component of our RAG pipeline is the LLM. It takes the prompt we built and generates an answer.

#### Sending the prompt to the LLM

We have the prompt from the previous section.

We send it to the LLM:


```python
response = openai_client.responses.create(
    model='gpt-5.4-mini',
    input=prompt
)
```

We use OpenAI's Responses API (```openai_client.responses.create```). OpenAI has two APIs: chat completions and responses. ```chat.completions``` is the older one, and it's now considered legacy. When the first edition of this course started, the ```responses``` API didn't exist, so we used ```chat.completions```. Now we prefer ```responses``` because it's more convenient.

There's a catch worth knowing. Many other providers like Groq and Gemini give you an OpenAI-compatible client. But they expose chat completions, not responses. So if you switch providers, you keep the OpenAI client but call ```chat.completions``` instead of ```responses```.

#### Exploring the response

The response is a Pydantic object. The answer is in response.output - a list of output items.

The first one is the message:


```python
response.output[0]
```




    ResponseOutputMessage(id='msg_05d778eaa59f0f6c006a35f990fd9c819984aa68fc9d25c5f5', content=[ResponseOutputText(annotations=[], text='Yes — you can still join and start learning now.\n\nIf you want a certificate, make sure to submit your project while submissions are still open.', type='output_text', logprobs=[])], role='assistant', status='completed', type='message', phase='final_answer')



The message has a content list, and the text is in the first item:


```python
response.output[0].content[0].text
```




    'Yes — you can still join and start learning now.\n\nIf you want a certificate, make sure to submit your project while submissions are still open.'



That's quite a journey to reach one string.

The shortcut spares us all of it:


```python
response.output_text
```




    'Yes — you can still join and start learning now.\n\nIf you want a certificate, make sure to submit your project while submissions are still open.'



Same result, less code. 

The usage counts tell you how many tokens the request consumed:


```python
response.usage
```




    ResponseUsage(input_tokens=480, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=39, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=519)



#### Calculating the price

You can use different models.

In this course we'll use gpt-5.4-mini:

* Input: $0.75 per million tokens
* Output: $4.50 per million tokens

Let's calculate the cost of the request we just made:


```python
input_price = 0.75 / 1_000_000
output_price = 4.50 / 1_000_000

cost = (
    response.usage.input_tokens * input_price +
    response.usage.output_tokens * output_price
)

cost
```




    0.0005355000000000001



This particular request costs a fraction of a cent. Even a full RAG query with a long prompt stays under $0.01. We need to send a lot of queries to even spend one cent. These models are cheap to play with.

The usage object also reports cached input tokens. Those are billed at a lower rate when the same prompt prefix repeats.

#### Message history

Previously we sent only one string as input and got back a response. In practice, you typically send a message history - a list of messages where each message has a role.

Think of a ChatGPT conversation. It starts with a hidden system prompt that tells the LLM how to behave, one you never see. After that, your messages and the LLM's replies alternate. The LLM has no memory of its own, so it needs the full history passed in to continue the conversation.

We won't build a multi-turn chat here. But we still use this message format to separate our instructions from the user prompt.

We send two messages:
* ```developer``` - system-level instructions (how the LLM should behave)
* ```user``` - the actual prompt with the question and context



```python
message_history = [
    {'role': 'developer', 'content': INSTRUCTIONS},
    {'role': 'user', 'content': prompt}
]

response = openai_client.responses.create(
    model='gpt-5.4-mini',
    input=message_history
)
```

This separates the fixed instructions from the user prompt, which changes every request.

OpenAI accepts both ```developer``` and ```system``` for the instruction role. There may be some difference between them, but in practice I don't see it change the result either way. We use ```developer``` in this course.

#### The LLM function

We can now put this together into an updated ```llm``` function.

It now takes both instructions and the user prompt:


```python
def llm(instructions, user_prompt, model='gpt-5.4-mini'):
    message_history = [
        {'role': 'developer', 'content': instructions},
        {'role': 'user', 'content': user_prompt}
    ]

    response = openai_client.responses.create(
        model=model,
        input=message_history
    )

    return response.output_text
```

#### Full RAG

With search, the prompt, and the LLM ready, we wire them together:


```python
def rag(query, model='gpt-5.4-mini'):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(INSTRUCTIONS, prompt, model=model)
    return answer
```

Try it:


```python
answer = rag("I just discovered the course. Can I join now?")
print(answer)
```

    Yes, but if you want to receive a certificate, you need to submit your project while submissions are still open.


The answer should be based on the FAQ documents - not on the LLM's general knowledge. The LLM read the search results and generated a response grounded in our data.

#### Try more questions

Try a few more:


```python
rag("How do I get a certificate?")
```




    'You can get a certificate only if you finish the course with a **live cohort**. Certificates are **not** awarded for the self-paced mode.\n\nTo be eligible, you need to **complete and pass the Capstone project**. You also need to **peer-review 3 capstones** during the course run, since peer review is required while the course is active.\n\nIf you want your real name on the certificate, update the **official name** field in your course profile.'



Notice how the answers reference specific courses and sections. The LLM reads from our knowledge base before answering; that's how RAG works.

This approach is modular. You can swap the search backend, the prompt template, or the LLM model. Nothing else needs to change. Later when we replace minsearch with sqlitesearch, only the ```search``` function changes.

[Back to TOC](#table-of-contents)

### RAG Helper


In the previous lessons, we built the RAG flow piece by piece - search, then the prompt, then the LLM call. The pipeline works, but every time we want to use it, we need to repeat the same code.

We'll use this code throughout the course, so let's put it into two reusable files:

* ingest.py - loading data and building the search index
* rag_helper.py - the RAG logic (search, prompt, LLM)

Then in notebooks, we just import from these files and use them.

#### ingest.py

This file handles data loading and index creation - everything we need before we can search.

Create ```ingest.py``` with two functions:


```python
import requests
from minsearch import Index

def load_faq_data():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw: 
        course_url = f"""{url_prefix}{course["path"]}"""
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents

def build_index(documents):
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"]
    )
    index.fit(documents)
    return index
```

We'll ```use load_faq_data()``` to fetch the documents and ```build_index()``` to create the minsearch index. Later, we'll add sqlitesearch support to this same file.

#### Using it in a notebook

Let's try it here in the notebook.  Import from both files and put everything together:


```python
from dotenv import load_dotenv
load_dotenv()

from ingest import load_faq_data, build_index
from rag_helper import RAGBase
from openai import OpenAI

documents = load_faq_data()
index = build_index(documents)

openai_client = OpenAI()

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
)

answer = assistant.rag("I just discovered the course. Can I join now?")
print(answer)
```

    Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions.


We can override the default instructions if we want:  


```python
custom_instructions = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
""".strip()

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
    instructions=custom_instructions,
)
```

Let's try a few more questions:


```python
assistant.rag("How do I get a certificate?")

```




    'You can get a certificate only if you finish the course with a **live cohort**. The course does **not** award certificates for the **self-paced** mode, because you need to **peer-review 3 capstone projects** after submitting your own project, and that can only happen while the course is running.\n\nIf you missed the first homework, that does **not** prevent you from getting a certificate — you still need to **pass the Capstone project**.'




```python
assistant.rag("Can I still join the course after it started?")
```




    'Yes, you can still join the course after it has started. You can start whenever you want.\n\nIf you want to receive a certificate, you need to finish the course with a live cohort and submit your project while submissions are still open.'



[Back to TOC](#table-of-contents)

### Persistent RAG


Up until this point, we've been using ```minsearch```, which is fine if our database is relatively small because the indexing is fast.  ```minsearch``` is in-memory; it is a bunch of python libraries bound to the process in which it is running.  Once you stop the process, the data disappears.  When you re-start the process, the indexing has to happen all over again.  This breaks down as the database grows, needlessly consuming time and resources.  

The solution to this is to separate the ingestion part of the process from querying.  One process writes the data to a persistent search index, while another process reads from it.  These two processes run independently, only sharing the index between them.   

There are several persistent search backend for this, such as Elasticsearch, OpenSearch, Qdrant, and ```sqlitesearch```.  In this module, we use ```sqlitesearch```, a library Alexey wrote.  It is a lightweight search library and has the same API as ```minsearch```, so we can easily drop it in to our code.  It leverages SQLite, which already ships with python, and puts an easier-to-use wrapper around python's full-text search engine.  

[Back to TOC](#table-of-contents)

### Demonstrating Persistent RAG in Action:  Exercise


Alexey instructs us to create two separate notebooks to demonstrate how the persistence process works.  One notebook ingests and indexes the document, creating an indexed database.  The other runs queries against that database.  That's how the two processes connect to each other.  

#### Ingestion Notebook

See https://github.com/gdurante2019/llm-zoomcamp-2026/blob/main/01-agentic-rag/sqlite-ingest.ipynb for the ingestion notebook.

#### Query notebook

See https://github.com/gdurante2019/llm-zoomcamp-2026/blob/main/01-agentic-rag/query-notebook.ipynb for the notebook executing the query.

[Back to TOC](#table-of-contents)

## Part 2:  Agents


In Part 1, we built a working RAG pipeline with keyword search from scratch.  This is a fixed pipeline that has three steps:  
1. Search the FAQ
2. Build a prompt with the results
3. Send it to the LLM for it to give a helpful reply.

It gives reasonably good results as long as the user's query matches text in the FAQ documents.  However, if the user's request contains a typo in a key word, or phrases the question in an unsual way, or requires information that could only be obtained through multiple searches, then this system breaks down.

Rather than building a prompt with the results of our search of the FAQ database and sending it to the LLM in the last step, we can put the LLM in charge of the search process.  With the LLM "in charge", it can:
* fix typos
* search again with different terms
* ask the use clarifying questions

Giving the LLM the ability to manage the process makes this system __*agentic*__.  An agent uses an LLM to decide which actions to take, and in what order.  Part 2 of this module covers key aspects of this structure, including:  
1. Function calling (giving the LLM the ability to use tools to solve the problem),
2. The agentic loop (in which the LLM decides when to call a tool, when to call another if needed, and when to stop and answer), and
3. Frameworks (the libraries that run this loop for us)

This part of the module builds on the RAG pipeline we constructed in Part 1.  


[Back to TOC](#table-of-contents)

### Quick RAG revision


As an example, we can see what happens when our query contains a typo.  We'll set up the RAG pipeline from Part 1 using our helper functions, ```ingest.py``` and ```rag_helper.py```.

First we need to load the OpenAI client.  We pull the necessary information, including the API key, from the environment file (this file is not committed to GitHub).  


```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()
```

Load the data and build the search index:


```python
from rag_helper import RAGBase
from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)
```

Create the assistant:


```python
instructions = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
""".strip()

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
    instructions=instructions,
)
```

#### Testing it

Let's try a question:


```python
assistant.rag("How do I run Ollama locally?")
```




    'To run Ollama locally:\n\n1. Install Ollama from https://ollama.com/download for your operating system:\n   - macOS: download the `.pkg`\n   - Windows: download the `.msi`\n   - Linux: run\n     ```bash\n     curl -fsSL https://ollama.com/install.sh | sh\n     ```\n\n2. In a terminal, start a model locally with:\n   ```bash\n   ollama run llama3\n   ```\n   This downloads the LLaMA 3 model, starts it locally, and opens a chat-like interface.\n\n3. To test the local Ollama server, run:\n   ```bash\n   curl http://localhost:11434\n   ```\n   You should get a response like:\n   ```json\n   {"models": [...]}  \n   ```\n\nIf you want to use it from Python, install the client with:\n```bash\npip install ollama\n```\n\nThen you can call it with `ollama.chat(...)`.'



We can see that this works fine, giving us the kind of answer we would expect.  But what is there is a typo in the request?


```python
assistant.rag("How do I run Olama locally?")
```




    'I don’t see any FAQ entry about running **Ollama** locally.\n\nThe closest relevant guidance in the FAQ is that you can run the course locally if you’re comfortable setting up the needed tools like Python, `uv`, Jupyter, Docker, and any other module-specific tools. If you want, I can help you figure out a local setup based on that.'



Because the term 'Olama' is not contained in the database, the LLM can't produce a result.  This is because the lexical search looks for the exact word.  This is where an agent can be really helpful.

[Back to TOC](#table-of-contents)

### The agentic approach


An agent puts the LLM in charge.

Instead of running search ourselves, we give the LLM a search tool. It decides when to call it and what to search for.

Now, instead of that exact question going into our pipeline and returning nothing, our pipeline process looks something like this:  
1. User question:  "How do I run Olama?"
2. LLM searches for 'Olama'
3. LLM turns up nothing in the database
4. LLM can infer that 'Olama' may actually refer to 'Ollama'
5. LLM runs the search again with "Ollama"
6. LLM returns helpful results!

The difference is about who makes the decisions.  

* With RAG, the developer decides. We fix the steps up front, so search always runs once with the exact user query.
* With an agent, the LLM decides. It chooses which actions to take and when to stop.

The mechanism that makes this possible is function calling, and that's what the rest of this lesson is about.

#### Asking without tools

First, let's see what the LLM does without any tools. We ask it a course-specific question and look at the answer.


```python
messages = [
    {"role": "user", "content": "I just discovered the course. Can I join it?"}
]

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
)

response.output_text
```




    'Yes—you can likely join it, but it depends on the course’s enrollment rules and whether registration is still open.\n\nIf you want, I can help you figure it out. Please send me:\n- the course name\n- the school/platform\n- when it starts\n- whether it’s in-person or online\n\nIf you’re asking the course organizer, you could say:\n> Hi, I just discovered the course and I’m very interested in joining. Is it still possible to enroll?'



Without any context, the LLM provides a genertic answer.  This is why we need RAG, and why we want to allow the model to use tools to provide useful results.

[Back to TOC](#table-of-contents)

### Defining the tools


First we define a top-level ```search``` function that queries ```index``` directly. The model will reference it by this name. We keep the Python function and the tool name aligned so the dispatch is easier later.


```python
def search(query):
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict
    )
```

Next we tell the model about this function. The model doesn't see our Python code, only a schema describing what the function does and what arguments it takes. LLMs are language agnostic. At the end we're just making an HTTP call, so we describe the tool in JSON rather than in Python. The same schema would work from TypeScript or Java.


```python
search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
```

The ```description``` is the most important field, because the model reads it to decide when to call the function. ```parameters``` is a JSON schema for the arguments, and we mark ```query``` as required so the model always fills it in.

#### Sending the question with the tool

Now we send the same question as before, but this time we include the tool in the request:


```python
response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

response.output
```




    [ResponseFunctionToolCall(arguments='{"query":"join course discovered late enrollment can I still join"}', call_id='call_y4RcKxRys2b6vWtwszpdfxb6', name='search', type='function_call', id='fc_068cdd426a095cfa006a39c32904bc819a9fa0f56131ad1761', namespace=None, status='completed')]



 Looking at the output, we see that the response contains a function_call entry. The model decided it needs to search the FAQ before answering. The models asks us to run the search function first.

Looking at the arguments too, we see something interesting.  The model didn't pass our question verbatim. It judged the raw question wasn't the best query to search with, so it rewrote our enrollment question into search keywords like "enroll late join course".

#### Executing the function and sending the result back

The function call contains JSON arguments. We parse them, call our ```search``` function, and serialize the result.


```python
import json

call = response.output[0]
args = json.loads(call.arguments)

results = search(**args)
result_json = json.dumps(results, indent=2)
```

Now we send this result back to the model. First, we add the model's output to the conversation history - the model needs to see its own function call. Then we add the tool result.


```python
messages.extend(response.output)

messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": result_json,
})
```

The ```call_id``` links the tool result to the specific function call the model requested. If the model makes multiple function calls in one turn, each one gets its own ```call_id```.

#### Asking the model again

We call the API a second time with the expanded history:


```python
response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

response.output_text
```




    'Yes — you can join even if you just discovered the course.\n\nIf you want a certificate, make sure to submit your project while submissions are still open.'



Hooray!  We get a DataTalksClub course-specific answer.  We see the original question, the model's decision to call ```search```, and the FAQ results.

Note that we have to send the whole history, because LLMs are stateless between API calls--meaning it doesn't 'remember' the previous information we might already have sent to it. The memory is the list you send as ```input```. If you send only the tool result, the model has no idea what's going on. So on this second call we replay everything we have so far. That means the question, the decision to call search, and the result we got back.

That's the full function-calling loop for a single turn. With plain RAG we made one call, and here we make two. Turning RAG agentic means more round-trips.

This pattern is referred to as "agentic RAG", "tool use", or "function calling". The idea behind all of them is the same: the LLM decides which tools to call.

#### Token usage and cost

We just made two API calls instead of one. Each call we send to the model costs money, so it's worth checking how much one tool-using turn actually costs.

The response has a usage field with the token counts:




```python
usage = response.usage
usage.input_tokens, usage.output_tokens
```




    (810, 35)



Model providers publish prices for each model per million input tokens and per million output tokens. Plug those numbers in to convert tokens to dollars.


```python
def calculate_gpt54mini_price(input_tokens, output_tokens):
    INPUT_PRICE_PER_MILLION = 0.15
    OUTPUT_PRICE_PER_MILLION = 0.60

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

result = calculate_gpt54mini_price(652, 33)
print("Total cost: $", round(result["total_cost"], 8))
```

    Total cost: $ 0.0001176


Note that this usage is only for the second API call. The first call also has its own usage and its own cost. That was the call where the model decided to invoke ```search```. Two calls means we pay twice. We pay even more on the second call, because we resend the full history as input.

With a real agent loop the model can make many calls, so the costs add up. It's a good idea to periodically check on ```usage``` as we develop apps.

[Back to TOC](#table-of-contents)

### The Agentic Loop


While this pipeline is an improvement on the fixed pipeline from Part 1, it only allows one function call--one iteration.  What if the first search misses the answer, or if the model would run more than one function call if it had the ability to do so?  To allow this, we need a loop that keeps calling the model and running tools until it's done.  This is what an agent does.


#### Anatomy of an agent

An LLM that is able to make decisions to best complete the task at hand is an agent. It's an AI assistant whose goal is to help the user.

An agent has three parts:

1.  *Instructions*, the role and behavior we want. We pass this as the ```developer``` message. The better the instructions, the better the agent helps.
2.  *Tools*, the functions the agent can call to carry out the task. For us that's only ```search```.
3.  *Memory*, the message history. We append every prompt, every model output, and every tool result. The agent reads this to know what it has already tried.

Everything below is the code that wires these three together inside a loop.

#### A developer prompt

So far we've relied on the model to figure out when to search. We can make this more reliable with a ```developer``` message that spells out how to conduct the task.  The same message also pushes it toward multiple searches, so we get to watch the loop run more than once.


```python
instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()
```

#### A function-call helper

We'll be running function calls repeatedly inside the loop, so let's wrap that in a small helper. It turns the JSON arguments into a Python dict, calls the right function, and serializes the result. We only have one tool for now, so we dispatch on the function name directly.


```python
def make_call(call):
    args = json.loads(call.arguments)

    if call.name == "search":
        result = search(**args)

    result_json = json.dumps(result, indent=2)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }
```

The helper returns the exact structure the Responses API expects. When we add more tools later, we'll extend this with more ```if``` branches (or switch to a registry).

#### Processing one response

Let's process a single model response. We append each output entry to the conversation, print any messages, and run any function calls. Function-call results get appended too.


```python
question = "I just discovered the course. Can I join it?"

messages = [
    {"role": "developer", "content": instructions},
    {"role": "user", "content": question},
]

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

messages.extend(response.output)
has_function_calls = False

for item in response.output:
    if item.type == "function_call":
        print("function_call:", item.name, item.arguments)
        call_output = make_call(item)
        messages.append(call_output)
        has_function_calls = True

    elif item.type == "message":
        print("ASSISTANT:")
        print(item.content[0].text)
```

    function_call: search {"query":"join course late enrollment discovered course can I join"}
    function_call: search {"query":"course enrollment can I still join if just discovered the course"}


The ```has_function_calls``` flag tells us whether the model needs another API call. If the response contains a function call, the updated ```messages``` has tool output the model hasn't seen yet. We'll need to send it back.

#### The full agent loop

We wrap this in a ```while``` loop.  The loop keeps calling the model until it returns a response without any function calls. We also keep an iteration counter so we can see how many round-trips happened.


```python
it = 1

while True:
    print(f"iteration #{it}...")
    has_function_calls = False

    response = openai_client.responses.create(
        model="gpt-5.4-mini",
        input=messages,
        tools=[search_tool],
    )

    messages.extend(response.output)

    for item in response.output:
        if item.type == "function_call":
            print("function_call:", item.name, item.arguments)
            call_output = make_call(item)
            messages.append(call_output)
            has_function_calls = True

        elif item.type == "message":
            print("ASSISTANT:")
            print(item.content[0].text)

    it = it + 1
    if has_function_calls == False:
        break
```

    iteration #1...
    ASSISTANT:
    Yes — you can still join the course after discovering it.
    
    A couple of important notes:
    - You can follow the materials and start learning anytime.
    - If you want a certificate, you need to submit your project while submissions are still open.
    - Certificates are only available for the live cohort, not self-paced study.
    
    If you want, I can also help you figure out the best way to catch up quickly. Are there other areas you’d like to explore?


This is the core agent loop. The model reasons about the next action.  The code performs it, and the model sees the result on the next turn.  The loop stops when the model returns a final answer with no more tool calls.

The model decides how many times it searches, and we keep looping until it stops asking for tools.

The exit condition is the simplest one possible.  If the model doesn't call a function in a turn, we're done.  Other frameworks add safety nets on top, like a max iteration count, a token budget, or a wall-clock limit. We could cap it at, say, five iterations and force an answer on the last one. The core is still this one flag.

#### Wrapping it in a function

Let's wrap the loop in a function so we can reuse it. The function takes the instructions and the question as parameters, and returns the final answer.


```python
def agent_loop(instructions, question, model="gpt-5.4-mini") -> str:
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question}
    ]

    it = 1

    while True:
        print(f"iteration #{it}...")
        has_function_calls = False

        response = openai_client.responses.create(
            model=model,
            input=messages,
            tools=[search_tool]
        )

        messages.extend(response.output)

        for item in response.output:
            if item.type == "function_call":
                print("function_call:", item.name, item.arguments)
                call_output = make_call(item)
                messages.append(call_output)
                has_function_calls = True

            elif item.type == "message":
                print("ASSISTANT:")
                last_answer = item.content[0].text
                print(item.content[0].text)

        it = it + 1
        if has_function_calls == False:
            break

    return last_answer
```

Let's try it with a question that has a typo:


```python
agent_loop(instructions, "How do I run Olama locally?")
```

    iteration #1...
    function_call: search {"query":"Ollama local run install start model pull run localhost FAQ"}
    function_call: search {"query":"run Ollama locally command ollama serve ollama run FAQ"}
    iteration #2...
    ASSISTANT:
    To run **Ollama locally**:
    
    1. **Install Ollama**
       - macOS: download the installer from https://ollama.com/download
       - Windows: download the `.msi`
       - Linux:
         ```bash
         curl -fsSL https://ollama.com/install.sh | sh
         ```
    
    2. **Start a model locally**
       ```bash
       ollama run llama3
       ```
       This will download the model and open a local chat interface.
    
    3. **Check that the local server is running**
       ```bash
       curl http://localhost:11434
       ```
       If it’s working, you should get a response from the Ollama server.
    
    4. **If you need to restart the server**
       ```bash
       nohup ollama serve > nohup.out 2>&1 &
       ```
    
    5. **Optional: use it from Python**
       ```bash
       pip install ollama
       ```
    
       ```python
       import ollama
    
       response = ollama.chat(
           model='llama3',
           messages=[{"role": "user", "content": "Hello!"}]
       )
    
       print(response['message']['content'])
       ```
    
    If you want, I can also show you how to use Ollama in a notebook or with RAG.





    'To run **Ollama locally**:\n\n1. **Install Ollama**\n   - macOS: download the installer from https://ollama.com/download\n   - Windows: download the `.msi`\n   - Linux:\n     ```bash\n     curl -fsSL https://ollama.com/install.sh | sh\n     ```\n\n2. **Start a model locally**\n   ```bash\n   ollama run llama3\n   ```\n   This will download the model and open a local chat interface.\n\n3. **Check that the local server is running**\n   ```bash\n   curl http://localhost:11434\n   ```\n   If it’s working, you should get a response from the Ollama server.\n\n4. **If you need to restart the server**\n   ```bash\n   nohup ollama serve > nohup.out 2>&1 &\n   ```\n\n5. **Optional: use it from Python**\n   ```bash\n   pip install ollama\n   ```\n\n   ```python\n   import ollama\n\n   response = ollama.chat(\n       model=\'llama3\',\n       messages=[{"role": "user", "content": "Hello!"}]\n   )\n\n   print(response[\'message\'][\'content\'])\n   ```\n\nIf you want, I can also show you how to use Ollama in a notebook or with RAG.'



We can see what happens at every step. The agent searches for "Olama" and doesn't get good results.  It then searches again with "Ollama" and finds the answer.  The loop allows the model recover from a bad search on its own. 

Also try the course enrollment question:


```python
agent_loop(instructions, "I just discovered the course. Can I still join it?")
```

    iteration #1...
    function_call: search {"query":"join course late enrollment discovered course can I still join"}
    function_call: search {"query":"course enrollment late join start date registration FAQ"}
    function_call: search {"query":"I just discovered the course can I still join FAQ"}
    iteration #2...
    ASSISTANT:
    Yes — you can still join the course. You can start learning and follow the materials even if you discovered it late.
    
    A couple of important notes:
    - If you want a certificate, you need to submit your project while submissions are still open.
    - If the course is currently running, you can also submit homework while the forms are open.
    
    If you want, I can also help you figure out the best way to catch up quickly.





    'Yes — you can still join the course. You can start learning and follow the materials even if you discovered it late.\n\nA couple of important notes:\n- If you want a certificate, you need to submit your project while submissions are still open.\n- If the course is currently running, you can also submit homework while the forms are open.\n\nIf you want, I can also help you figure out the best way to catch up quickly.'



We get good results here as well:  DataTalksClub-specific course information that answers the question.

[Back to TOC](#table-of-contents)

### Encouraging multiple searches


There's a subtle issue here. The model often answers after the first search, even when more searches would help. It reasons that it already knows enough, so it doesn't run additional searches.  Let's push it to explore more by rewriting the instructions.


```python
instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results 
and then perform more searches. 

At the end, ask if there are other areas that the user wants to explore.
""".strip()

agent_loop(instructions, "I just discovered the course. Can I join it?")
```

    iteration #1...
    function_call: search {"query":"join course late enrollment discovered course can I join FAQ"}
    iteration #2...
    function_call: search {"query":"certificate submit project while accepting submissions peer review live cohort self-paced course FAQ"}
    iteration #3...
    ASSISTANT:
    Yes — you can still join the course.
    
    A couple of important notes:
    - You can start learning and following the materials even if you discovered it late.
    - If you want a certificate, you need to submit your project while submissions are still open, and the course certificate is only available for the live cohort, not self-paced.
    
    If you want, I can also help you figure out how to start catching up quickly. Are there other areas you want to explore?





    'Yes — you can still join the course.\n\nA couple of important notes:\n- You can start learning and following the materials even if you discovered it late.\n- If you want a certificate, you need to submit your project while submissions are still open, and the course certificate is only available for the live cohort, not self-paced.\n\nIf you want, I can also help you figure out how to start catching up quickly. Are there other areas you want to explore?'



Now the agent makes multiple searches per question and doesn't stop after the first round of results. The **_instructions_** are how we steer the agent. It can still decide to skip ahead, though, so don't expect it to follow them every single run.

#### Restricting off-topic questions

Right now the agent will try to answer anything we ask it.  For example, we could ask it a question about chess and it will attempt to answer:


```python
agent_loop(instructions, "What is the Queen's Gambit?")
```

    iteration #1...
    function_call: search {"query":"Queen's Gambit definition chess opening course FAQ"}
    iteration #2...
    function_call: search {"query":"Queen's Gambit chess opening what is it explanation"}
    iteration #3...
    ASSISTANT:
    The Queen’s Gambit is a **chess opening** that starts with the moves:
    
    1. d4 d5  
    2. c4
    
    In this opening, White offers a pawn on c4 to try to **gain control of the center** and create a more active position. If Black accepts the pawn, it’s called the **Queen’s Gambit Accepted**; if Black doesn’t, it’s the **Queen’s Gambit Declined**.
    
    If you want, I can also explain:
    - why it’s called a “gambit,”
    - the main ideas for White and Black,
    - or show a few common lines.





    'The Queen’s Gambit is a **chess opening** that starts with the moves:\n\n1. d4 d5  \n2. c4\n\nIn this opening, White offers a pawn on c4 to try to **gain control of the center** and create a more active position. If Black accepts the pawn, it’s called the **Queen’s Gambit Accepted**; if Black doesn’t, it’s the **Queen’s Gambit Declined**.\n\nIf you want, I can also explain:\n- why it’s called a “gambit,”\n- the main ideas for White and Black,\n- or show a few common lines.'



We don't want to expend resources answering questions that don't pertain to our courses, so we can modify our instructions to the LLM to only answer questions from the course FAQ.  


```python
instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results 
and then perform more searches. 

The question has to be about the course or its logistics, offtopic questions 
shouldn't be answered. If the search returns nothing, it's likely an off-topic question.
If you can't answer the question using FAQ, don't do it yourself. Only use the 
facts from the FAQ database.

At the end, ask if there are other areas that the user wants to explore.
""".strip()

agent_loop(instructions, "what's queen gambit?")
```

    iteration #1...
    function_call: search {"query":"queen gambit chess opening queen gambit"}
    iteration #2...
    function_call: search {"query":"queen's gambit queen gambit course faq"}
    iteration #3...
    ASSISTANT:
    I couldn’t find any course FAQ entry about “queen gambit,” so this seems off-topic for the course.
    
    If you meant something else related to the course, feel free to clarify. Otherwise, is there another course-related area you want to explore?





    'I couldn’t find any course FAQ entry about “queen gambit,” so this seems off-topic for the course.\n\nIf you meant something else related to the course, feel free to clarify. Otherwise, is there another course-related area you want to explore?'



This is a lightweight form of an input guardrail:  we tell the agent what's in scope and what isn't.  A *real* guardrail checks the input before the agent runs and can block off-topic questions outright. While we aren't looking to implement real guardrails right now, we see that instructions are the first place to start with any kind of guardrail.

This handwritten loop is the best way to understand what frameworks don't automatically reveal. Every agent framework wraps this same pattern, whether it's LangChain, PydanticAI, or the OpenAI Agents SDK.

[Back to TOC](#table-of-contents)

### ToyAIKit


Writing the agent loop by hand is educational but tedious.  We don't want to have to do this every time.  We'd like to have a toolkit that can quickly do this for us, so we can focus on prompts, tools, and behaviors.  

Alexey built ToyAIKit in a DTC workshop a while back.  It's small and easy to read, so it's useful for developing and debugging locally and learning the basics in this course.  Note:  ToyAIKit is an experimental library built for educational purposes, and is not intended for use in production.  

#### Setup

Install it:


```python
# !uv add toyaikit
```


Import the classes we need:


```python
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback
```

#### Registering the tool

We register our ```search``` function along with the schema from earlier lessons:




```python
agent_tools = Tools()
agent_tools.add_tool(search, search_tool)
```

#### Letting ToyAIKit generate the schema

Writing that schema by hand is annoying, and we don't want to do it for every function.  Luckily, we don't have to.

If we add a type hint and a docstring to ```search```, ToyAIKit reads them and derives the schema for us:


```python
def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )
```

Then register it without passing a schema:


```python
agent_tools = Tools()
agent_tools.add_tool(search)
```

Let's see what ToyAIKit produced:


```python
agent_tools.get_tools()
```




    [{'type': 'function',
      'name': 'search',
      'description': 'Search the FAQ database for entries matching the given query.',
      'parameters': {'type': 'object',
       'properties': {'query': {'type': 'string',
         'description': 'query parameter'}},
       'required': ['query'],
       'additionalProperties': False}}]



The output is the same JSON schema we hand-wrote in the function calling lesson. ToyAIKit generated it from the docstring and the type hint.

Every modern agent framework does this same trick. It reads a typed Python function with a docstring and builds the schema from it. The OpenAI Agents SDK, PydanticAI, LangChain and Google ADK all work this way. You write the tool and the framework figures out how to describe it.



#### The chat interface and runner

Create the chat interface and a callback, then build the runner:


```python
chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)
```

The ```chat_interface``` handles display in the notebook. The ```callback``` renders model messages and tool calls as they happen. The ```runner``` runs the agent loop, the same ```while True``` we wrote by hand. It sends messages, executes function calls, adds tool outputs back, and repeats until the model is done.

We pick ```gpt-5.4-mini``` here on purpose. Without it, ToyAIKit falls back to a smaller, faster default that doesn't follow the instructions as reliably.



#### Running one prompt
Run a single prompt:


```python
result = runner.loop(
    prompt="How do I run Olama locally?",
    callback=callback,
)
```

    -> Response received




            <details>
            <summary>Function call: <tt>search({"query":"Olama local run Ollama locally"})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{"query":"Olama local run Ollama locally"}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>[
  {
    "id": "1d0b969028",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Ollama: How to install Ollama?",
    "answer": "First, install Ollama by visiting [https://ollama.com/download](https://ollama.com/download) and choosing your operating system:\n\n- **macOS**: Download the `.pkg` and install it.\n- **Windows**: Download the `.msi` and install it.\n- **Linux**: Run the following command in the terminal:\n\n  ```bash\n  curl -fsSL https://ollama.com/install.sh | sh\n  ```\n\nOnce installed, open a terminal and type:\n\n```bash\nollama run llama3\n```\n\nThis command will:\n\n- Download the LLaMA 3 model (~4GB).\n- Start the model locally.\n- Open a chat-like interface where you can type questions.\n\nTo test the Ollama local server, run the following command:\n\n```bash\ncurl http://localhost:11434\n```\n\nYou should receive a response similar to:\n\n```json\n{\"models\": [...]}  \n```\n\nThen, install the Python client with:\n\n```bash\npip install ollama\n```\n\nHere is a minimal Python example:\n\n```python\nimport ollama\n\nresponse = ollama.chat(\n    model='llama3',\n    messages=[{\"role\": \"user\", \"content\": your_prompt}]\n)\n\nprint(response['message']['content'])\n```"
  },
  {
    "id": "aa310de435",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Can I run the course locally instead of Codespaces?",
    "answer": "Yes. Codespaces is just the easiest way for everyone to start with the same environment.\n\nYou can run the course locally if you are comfortable setting up Python, `uv`, Jupyter, Docker, and any other tools needed for the module.\n\nIf you run locally, make sure you document your setup and keep your environment reproducible."
  },
  {
    "id": "c6fc2d4d11",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Connection refused error when prompting Ollama RAG",
    "answer": "If you encounter this error while doing the homework, you can resolve it by restarting the Ollama server using the following command:\n\n```bash\n!nohup ollama serve &gt; nohup.out 2&gt;&1 &\n```\n\nMake sure to rerun the cell containing `ollama serve` if you stop and restart the notebook cell."
  },
  {
    "id": "15e931476c",
    "course": "llm-zoomcamp",
    "section": "Module 5: Monitoring",
    "question": "In Windows OS: OSError: [WinError 126] The specified module could not be found. Error loading \"C:\\Users\\USER\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\torch\\lib\\fbgemm.dll\" or one of its dependencies.",
    "answer": "- **Solution 1**: Install Visual C++ Redistributable.\n\n- **Solution 2**: Install Visual Studio, not Visual Studio Code. \n\n  &lt;{IMAGE:image_1}&gt;\n\n  For more details, please follow this link: [discuss.pytorch.org](https://discuss.pytorch.org/t/failed-to-import-pytorch-fbgemm-dll-or-one-of-its-dependencies-is-missing/201969)"
  },
  {
    "id": "0d74a3616f",
    "course": "llm-zoomcamp",
    "section": "Module 1: Agentic RAG",
    "question": "Any free models with tool use support?",
    "answer": "Several Groq models offer tool use, such as Deepseek R1 or Llama 4, all of which can be used for free for development.\n\nOther providers also support tool or function calling, including Mistral, Gemini, and some local Ollama models.\n\nYou'll typically need to adapt the code when not using OpenAI, because tool schemas and response shapes differ between providers.\n\nFor more details, see the [Groq Tool Use Documentation](https://console.groq.com/docs/tool-use)."
  }
]</pre>
            </div>

            </details>



    -> Response received




            <details>
            <summary>Function call: <tt>search({"query":"Ollama run llama3 localhost 11434 ser...)</tt></summary>
            <div>
                <b>Call</b>
                <pre>{"query":"Ollama run llama3 localhost 11434 serve connection refused ollama locally install"}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>[
  {
    "id": "1d0b969028",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Ollama: How to install Ollama?",
    "answer": "First, install Ollama by visiting [https://ollama.com/download](https://ollama.com/download) and choosing your operating system:\n\n- **macOS**: Download the `.pkg` and install it.\n- **Windows**: Download the `.msi` and install it.\n- **Linux**: Run the following command in the terminal:\n\n  ```bash\n  curl -fsSL https://ollama.com/install.sh | sh\n  ```\n\nOnce installed, open a terminal and type:\n\n```bash\nollama run llama3\n```\n\nThis command will:\n\n- Download the LLaMA 3 model (~4GB).\n- Start the model locally.\n- Open a chat-like interface where you can type questions.\n\nTo test the Ollama local server, run the following command:\n\n```bash\ncurl http://localhost:11434\n```\n\nYou should receive a response similar to:\n\n```json\n{\"models\": [...]}  \n```\n\nThen, install the Python client with:\n\n```bash\npip install ollama\n```\n\nHere is a minimal Python example:\n\n```python\nimport ollama\n\nresponse = ollama.chat(\n    model='llama3',\n    messages=[{\"role\": \"user\", \"content\": your_prompt}]\n)\n\nprint(response['message']['content'])\n```"
  },
  {
    "id": "c6fc2d4d11",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Connection refused error when prompting Ollama RAG",
    "answer": "If you encounter this error while doing the homework, you can resolve it by restarting the Ollama server using the following command:\n\n```bash\n!nohup ollama serve &gt; nohup.out 2&gt;&1 &\n```\n\nMake sure to rerun the cell containing `ollama serve` if you stop and restart the notebook cell."
  },
  {
    "id": "aa310de435",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Can I run the course locally instead of Codespaces?",
    "answer": "Yes. Codespaces is just the easiest way for everyone to start with the same environment.\n\nYou can run the course locally if you are comfortable setting up Python, `uv`, Jupyter, Docker, and any other tools needed for the module.\n\nIf you run locally, make sure you document your setup and keep your environment reproducible."
  },
  {
    "id": "0bed1f48da",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "dotenv is not recognized. What should I install?",
    "answer": "Install `python-dotenv`:\n\n```bash\nuv add python-dotenv\n```\n\nThen import and use it in Python:\n\n```python\nfrom dotenv import load_dotenv\n\nload_dotenv()\n```\n\nThe package is documented here: [python-dotenv](https://pypi.org/project/python-dotenv/)."
  },
  {
    "id": "0d74a3616f",
    "course": "llm-zoomcamp",
    "section": "Module 1: Agentic RAG",
    "question": "Any free models with tool use support?",
    "answer": "Several Groq models offer tool use, such as Deepseek R1 or Llama 4, all of which can be used for free for development.\n\nOther providers also support tool or function calling, including Mistral, Gemini, and some local Ollama models.\n\nYou'll typically need to adapt the code when not using OpenAI, because tool schemas and response shapes differ between providers.\n\nFor more details, see the [Groq Tool Use Documentation](https://console.groq.com/docs/tool-use)."
  }
]</pre>
            </div>

            </details>



    -> Response received




            <div>
                <div><b>Assistant:</b></div>
                <div><p>To run Ollama locally:</p>
<ol>
<li><p>Install Ollama from: https://ollama.com/download</p>
<ul>
<li>macOS: download the <code>.pkg</code></li>
<li>Windows: download the <code>.msi</code></li>
<li>Linux: run:</li>
</ul>
<pre><code class="language-bash">curl -fsSL https://ollama.com/install.sh | sh
</code></pre>
</li>
<li><p>Start a model locally:</p>
<pre><code class="language-bash">ollama run llama3
</code></pre>
<p>This will download the model and open a chat-like interface.</p>
</li>
<li><p>To test that the local Ollama server is running:</p>
<pre><code class="language-bash">curl http://localhost:11434
</code></pre>
</li>
</ol>
<p>If you get a connection refused error, restart the Ollama server with:</p>
<pre><code class="language-bash">!nohup ollama serve &gt; nohup.out 2&gt;&amp;1 &amp;
</code></pre>
<p>If you want to use it from Python:</p>
<pre><code class="language-bash">pip install ollama
</code></pre>
<pre><code class="language-python">import ollama

response = ollama.chat(
    model='llama3',
    messages=[{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: your_prompt}]
)

print(response['message']['content'])
</code></pre>
<p>Would you like to explore anything else?</p>
</div>
            </div>



We used the typo "Olama" on purpose. The agent searches and gets poor results, then retries with "Ollama". The recovery is the same as the handwritten loop. The notebook output is nicer to watch. Each tool call and message renders inline, so you can look at every search result.

The ```result``` is a ```LoopResult``` with ```all_messages``` (the full conversation), token counts, and ```cost``` (computed from token usage).



#### Cost and tokens
Let's find out what the call cost:


```python
result.cost
```




    CostInfo(input_cost=Decimal('0.0027555'), output_cost=Decimal('0.0013365'), total_cost=Decimal('0.0040920'))



This is useful while developing--especially with multi-turn agents where one prompt can trigger several model calls. The handwritten loop made you compute this by hand. The framework keeps a running total for you.

You can also look at the full message history:


```python
result.all_messages
```




    [EasyInputMessage(content="You're a course teaching assistant.\nYou're given a question from a course student and your task is to answer it.\n\nIf you want to look up information, use the search function. \nUse as many keywords from the user question as possible when making first requests.\n\nMake multiple searches. First perform search, analyze the results \nand then perform more searches. \n\nThe question has to be about the course or its logistics, offtopic questions \nshouldn't be answered. If the search returns nothing, it's likely an off-topic question.\nIf you can't answer the question using FAQ, don't do it yourself. Only use the \nfacts from the FAQ database.\n\nAt the end, ask if there are other areas that the user wants to explore.", role='developer', phase=None, type=None),
     EasyInputMessage(content='How do I run Olama locally?', role='user', phase=None, type=None),
     ResponseFunctionToolCall(arguments='{"query":"Olama local run Ollama locally"}', call_id='call_j6KPpUT1htYfnvrmsDUbLJz2', name='search', type='function_call', id='fc_0d93c51d0ce7e4b3006a39c34016d481989d271cc9a70a0378', namespace=None, status='completed'),
     {'type': 'function_call_output',
      'call_id': 'call_j6KPpUT1htYfnvrmsDUbLJz2',
      'output': '[\n  {\n    "id": "1d0b969028",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: RAG",\n    "question": "Ollama: How to install Ollama?",\n    "answer": "First, install Ollama by visiting [https://ollama.com/download](https://ollama.com/download) and choosing your operating system:\\n\\n- **macOS**: Download the `.pkg` and install it.\\n- **Windows**: Download the `.msi` and install it.\\n- **Linux**: Run the following command in the terminal:\\n\\n  ```bash\\n  curl -fsSL https://ollama.com/install.sh | sh\\n  ```\\n\\nOnce installed, open a terminal and type:\\n\\n```bash\\nollama run llama3\\n```\\n\\nThis command will:\\n\\n- Download the LLaMA 3 model (~4GB).\\n- Start the model locally.\\n- Open a chat-like interface where you can type questions.\\n\\nTo test the Ollama local server, run the following command:\\n\\n```bash\\ncurl http://localhost:11434\\n```\\n\\nYou should receive a response similar to:\\n\\n```json\\n{\\"models\\": [...]}  \\n```\\n\\nThen, install the Python client with:\\n\\n```bash\\npip install ollama\\n```\\n\\nHere is a minimal Python example:\\n\\n```python\\nimport ollama\\n\\nresponse = ollama.chat(\\n    model=\'llama3\',\\n    messages=[{\\"role\\": \\"user\\", \\"content\\": your_prompt}]\\n)\\n\\nprint(response[\'message\'][\'content\'])\\n```"\n  },\n  {\n    "id": "aa310de435",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: RAG",\n    "question": "Can I run the course locally instead of Codespaces?",\n    "answer": "Yes. Codespaces is just the easiest way for everyone to start with the same environment.\\n\\nYou can run the course locally if you are comfortable setting up Python, `uv`, Jupyter, Docker, and any other tools needed for the module.\\n\\nIf you run locally, make sure you document your setup and keep your environment reproducible."\n  },\n  {\n    "id": "c6fc2d4d11",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: RAG",\n    "question": "Connection refused error when prompting Ollama RAG",\n    "answer": "If you encounter this error while doing the homework, you can resolve it by restarting the Ollama server using the following command:\\n\\n```bash\\n!nohup ollama serve > nohup.out 2>&1 &\\n```\\n\\nMake sure to rerun the cell containing `ollama serve` if you stop and restart the notebook cell."\n  },\n  {\n    "id": "15e931476c",\n    "course": "llm-zoomcamp",\n    "section": "Module 5: Monitoring",\n    "question": "In Windows OS: OSError: [WinError 126] The specified module could not be found. Error loading \\"C:\\\\Users\\\\USER\\\\AppData\\\\Local\\\\Programs\\\\Python\\\\Python310\\\\lib\\\\site-packages\\\\torch\\\\lib\\\\fbgemm.dll\\" or one of its dependencies.",\n    "answer": "- **Solution 1**: Install Visual C++ Redistributable.\\n\\n- **Solution 2**: Install Visual Studio, not Visual Studio Code. \\n\\n  <{IMAGE:image_1}>\\n\\n  For more details, please follow this link: [discuss.pytorch.org](https://discuss.pytorch.org/t/failed-to-import-pytorch-fbgemm-dll-or-one-of-its-dependencies-is-missing/201969)"\n  },\n  {\n    "id": "0d74a3616f",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: Agentic RAG",\n    "question": "Any free models with tool use support?",\n    "answer": "Several Groq models offer tool use, such as Deepseek R1 or Llama 4, all of which can be used for free for development.\\n\\nOther providers also support tool or function calling, including Mistral, Gemini, and some local Ollama models.\\n\\nYou\'ll typically need to adapt the code when not using OpenAI, because tool schemas and response shapes differ between providers.\\n\\nFor more details, see the [Groq Tool Use Documentation](https://console.groq.com/docs/tool-use)."\n  }\n]'},
     ResponseFunctionToolCall(arguments='{"query":"Ollama run llama3 localhost 11434 serve connection refused ollama locally install"}', call_id='call_kTHU2T0bFgzzGjdM2t4VmpgA', name='search', type='function_call', id='fc_0d93c51d0ce7e4b3006a39c34151a88198a0792dcee2f0a1bc', namespace=None, status='completed'),
     {'type': 'function_call_output',
      'call_id': 'call_kTHU2T0bFgzzGjdM2t4VmpgA',
      'output': '[\n  {\n    "id": "1d0b969028",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: RAG",\n    "question": "Ollama: How to install Ollama?",\n    "answer": "First, install Ollama by visiting [https://ollama.com/download](https://ollama.com/download) and choosing your operating system:\\n\\n- **macOS**: Download the `.pkg` and install it.\\n- **Windows**: Download the `.msi` and install it.\\n- **Linux**: Run the following command in the terminal:\\n\\n  ```bash\\n  curl -fsSL https://ollama.com/install.sh | sh\\n  ```\\n\\nOnce installed, open a terminal and type:\\n\\n```bash\\nollama run llama3\\n```\\n\\nThis command will:\\n\\n- Download the LLaMA 3 model (~4GB).\\n- Start the model locally.\\n- Open a chat-like interface where you can type questions.\\n\\nTo test the Ollama local server, run the following command:\\n\\n```bash\\ncurl http://localhost:11434\\n```\\n\\nYou should receive a response similar to:\\n\\n```json\\n{\\"models\\": [...]}  \\n```\\n\\nThen, install the Python client with:\\n\\n```bash\\npip install ollama\\n```\\n\\nHere is a minimal Python example:\\n\\n```python\\nimport ollama\\n\\nresponse = ollama.chat(\\n    model=\'llama3\',\\n    messages=[{\\"role\\": \\"user\\", \\"content\\": your_prompt}]\\n)\\n\\nprint(response[\'message\'][\'content\'])\\n```"\n  },\n  {\n    "id": "c6fc2d4d11",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: RAG",\n    "question": "Connection refused error when prompting Ollama RAG",\n    "answer": "If you encounter this error while doing the homework, you can resolve it by restarting the Ollama server using the following command:\\n\\n```bash\\n!nohup ollama serve > nohup.out 2>&1 &\\n```\\n\\nMake sure to rerun the cell containing `ollama serve` if you stop and restart the notebook cell."\n  },\n  {\n    "id": "aa310de435",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: RAG",\n    "question": "Can I run the course locally instead of Codespaces?",\n    "answer": "Yes. Codespaces is just the easiest way for everyone to start with the same environment.\\n\\nYou can run the course locally if you are comfortable setting up Python, `uv`, Jupyter, Docker, and any other tools needed for the module.\\n\\nIf you run locally, make sure you document your setup and keep your environment reproducible."\n  },\n  {\n    "id": "0bed1f48da",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: RAG",\n    "question": "dotenv is not recognized. What should I install?",\n    "answer": "Install `python-dotenv`:\\n\\n```bash\\nuv add python-dotenv\\n```\\n\\nThen import and use it in Python:\\n\\n```python\\nfrom dotenv import load_dotenv\\n\\nload_dotenv()\\n```\\n\\nThe package is documented here: [python-dotenv](https://pypi.org/project/python-dotenv/)."\n  },\n  {\n    "id": "0d74a3616f",\n    "course": "llm-zoomcamp",\n    "section": "Module 1: Agentic RAG",\n    "question": "Any free models with tool use support?",\n    "answer": "Several Groq models offer tool use, such as Deepseek R1 or Llama 4, all of which can be used for free for development.\\n\\nOther providers also support tool or function calling, including Mistral, Gemini, and some local Ollama models.\\n\\nYou\'ll typically need to adapt the code when not using OpenAI, because tool schemas and response shapes differ between providers.\\n\\nFor more details, see the [Groq Tool Use Documentation](https://console.groq.com/docs/tool-use)."\n  }\n]'},
     ResponseOutputMessage(id='msg_0d93c51d0ce7e4b3006a39c3428de08198a1737a80b1d80060', content=[ResponseOutputText(annotations=[], text='To run Ollama locally:\n\n1. Install Ollama from: https://ollama.com/download  \n   - macOS: download the `.pkg`\n   - Windows: download the `.msi`\n   - Linux: run:\n   ```bash\n   curl -fsSL https://ollama.com/install.sh | sh\n   ```\n\n2. Start a model locally:\n   ```bash\n   ollama run llama3\n   ```\n   This will download the model and open a chat-like interface.\n\n3. To test that the local Ollama server is running:\n   ```bash\n   curl http://localhost:11434\n   ```\n\nIf you get a connection refused error, restart the Ollama server with:\n```bash\n!nohup ollama serve > nohup.out 2>&1 &\n```\n\nIf you want to use it from Python:\n```bash\npip install ollama\n```\n\n```python\nimport ollama\n\nresponse = ollama.chat(\n    model=\'llama3\',\n    messages=[{"role": "user", "content": your_prompt}]\n)\n\nprint(response[\'message\'][\'content\'])\n```\n\nWould you like to explore anything else?', type='output_text', logprobs=[])], role='assistant', status='completed', type='message', phase='final_answer')]



This is just a list - the same ```messages``` list we maintained by hand.

#### Continuing the conversation
Take the messages from the previous result and pass them as ```previous_messages``` on the next ```loop``` call:


```python
result2 = runner.loop(
    prompt="How do I run a different model?",
    previous_messages=result.all_messages,
    callback=callback,
)
```

    -> Response received




            <details>
            <summary>Function call: <tt>search({"query":"Ollama run different model model name...)</tt></summary>
            <div>
                <b>Call</b>
                <pre>{"query":"Ollama run different model model name llama3 qwen mistral course FAQ"}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>[
  {
    "id": "1d0b969028",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Ollama: How to install Ollama?",
    "answer": "First, install Ollama by visiting [https://ollama.com/download](https://ollama.com/download) and choosing your operating system:\n\n- **macOS**: Download the `.pkg` and install it.\n- **Windows**: Download the `.msi` and install it.\n- **Linux**: Run the following command in the terminal:\n\n  ```bash\n  curl -fsSL https://ollama.com/install.sh | sh\n  ```\n\nOnce installed, open a terminal and type:\n\n```bash\nollama run llama3\n```\n\nThis command will:\n\n- Download the LLaMA 3 model (~4GB).\n- Start the model locally.\n- Open a chat-like interface where you can type questions.\n\nTo test the Ollama local server, run the following command:\n\n```bash\ncurl http://localhost:11434\n```\n\nYou should receive a response similar to:\n\n```json\n{\"models\": [...]}  \n```\n\nThen, install the Python client with:\n\n```bash\npip install ollama\n```\n\nHere is a minimal Python example:\n\n```python\nimport ollama\n\nresponse = ollama.chat(\n    model='llama3',\n    messages=[{\"role\": \"user\", \"content\": your_prompt}]\n)\n\nprint(response['message']['content'])\n```"
  },
  {
    "id": "c6fc2d4d11",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Connection refused error when prompting Ollama RAG",
    "answer": "If you encounter this error while doing the homework, you can resolve it by restarting the Ollama server using the following command:\n\n```bash\n!nohup ollama serve &gt; nohup.out 2&gt;&1 &\n```\n\nMake sure to rerun the cell containing `ollama serve` if you stop and restart the notebook cell."
  },
  {
    "id": "ee43413718",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Can I use a model or provider different from the one recommended in homework?",
    "answer": "Yes. The recommended model is not mandatory. You can use OpenAI, Gemini, Groq, OpenRouter, Azure OpenAI, local models, or another provider.\n\nThe homework is designed so you do not need a paid service. You may need to adapt the code for your provider, because response formats, tool schemas, and tokenizers differ.\n\nFor provider ideas, see the course list of [OpenAI API alternatives](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/awesome-llms.md)."
  },
  {
    "id": "aa310de435",
    "course": "llm-zoomcamp",
    "section": "Module 1: RAG",
    "question": "Can I run the course locally instead of Codespaces?",
    "answer": "Yes. Codespaces is just the easiest way for everyone to start with the same environment.\n\nYou can run the course locally if you are comfortable setting up Python, `uv`, Jupyter, Docker, and any other tools needed for the module.\n\nIf you run locally, make sure you document your setup and keep your environment reproducible."
  },
  {
    "id": "fe8fed31e6",
    "course": "llm-zoomcamp",
    "section": "Module 1 Homework",
    "question": "How do I get token counts for Module 1 homework if I use a different provider?",
    "answer": "For the current Module 1 homework, get the token count from the model response object.\n\nFor example, OpenAI-compatible clients usually return usage information on the response, such as `response.usage.input_tokens` or `response.usage.prompt_tokens`, depending on the API style.\n\nIf you use a non-OpenAI provider, check the provider's response object for its usage fields and adapt the code. Do not use `tiktoken` or `cl100k_base` as a generic tokenizer for Gemini, Mistral, Hugging Face, Groq, or other providers because tokenization differs by model.\n\nIf your provider does not expose token usage, use that provider's native tokenizer as an approximation. For multiple-choice homework questions, choose the closest option."
  }
]</pre>
            </div>

            </details>



    -> Response received




            <div>
                <div><b>Assistant:</b></div>
                <div><p>You can run a different model by giving <code>ollama run</code> a different model name, for example:</p>
<pre><code class="language-bash">ollama run llama3
</code></pre>
<p>The FAQ only explicitly shows <code>llama3</code>, but it also says you can use local models or a different provider if needed. If you use another model/provider, you may need to adapt the code because response formats and tokenization can differ.</p>
<p>If you’re using Python, change the model name here too:</p>
<pre><code class="language-python">response = ollama.chat(
    model='llama3',
    messages=[{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: your_prompt}]
)
</code></pre>
<p>Replace <code>'llama3'</code> with the model you want to use.</p>
<p>Would you like to explore anything else?</p>
</div>
            </div>



The runner picks up where the last call left off, with the same agent loop and an extended history. The model knows "different model" refers to Ollama because it sees the previous turn in memory. Without that history, it would have no idea what we're asking about.

#### Interactive chat
For a chat-like workflow, run the built-in input loop, typing questions in the chat box that appears on screen.  (To exit the interactive chat, type "stop".)


```python
runner.run();

# In VS Code, the interactive chat window is at the top of the Jupyter Notebook window.

```

    -> Response received




            <details>
            <summary>Function call: <tt>search({"query":"course logistics FAQ schedule assignm...)</tt></summary>
            <div>
                <b>Call</b>
                <pre>{"query":"course logistics FAQ schedule assignments office hours exam project submission"}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>[
  {
    "id": "d65e05bd7a",
    "course": "llm-zoomcamp",
    "section": "General Course-Related Questions",
    "question": "Are there live sessions or office hours for each module?",
    "answer": "There are no separate live sessions for every module by default. Module materials are pre-recorded and available in the course repo.\n\nLive sessions are announced separately when they happen. If you are stuck, ask your question in Slack and follow the [asking questions guidelines](https://datatalks.club/docs/courses/zoomcamp-logistics/asking-questions/).\n\nOptional extra support is available through [AI Shipping Labs](https://datatalks.club/docs/courses/llm-zoomcamp/ai-shipping-labs/), a paid community that includes regular Zoom office hours and additional structure. This is optional; the DataTalks.Club course content remains free."
  },
  {
    "id": "489dd1c9d9",
    "course": "llm-zoomcamp",
    "section": "General Course-Related Questions",
    "question": "What is the video/zoom link to the stream for the \u201cOffice Hours\u201d or live/workshop sessions?",
    "answer": "The zoom link is only published to instructors/presenters/TAs.\n\nStudents participate via YouTube Live and submit questions to Slido (link is pinned in the chat when live). The video URL should be posted in the [announcements channel on Telegram and Slack](https://t.me/dezoomcamp) before it begins. You can also watch live on the DataTalksClub [YouTube Channel](https://www.youtube.com/c/DataTalksClub).\n\nDon\u2019t post questions in chat as they may be missed if the room is very active."
  },
  {
    "id": "0fab61eca2",
    "course": "llm-zoomcamp",
    "section": "Capstone Project",
    "question": "Is it a group project?",
    "answer": "No, the capstone is an individual project.\n\nYou can collaborate or discuss a larger idea with other students, but each submitted project must stand on its own. A shared system can work only if it is clearly decomposed into independent projects, where each person has a separate qualifying component and a separate repository.\n\nIf the work cannot be evaluated independently for each participant, it does not satisfy the project requirement."
  },
  {
    "id": "a8a7fef016",
    "course": "llm-zoomcamp",
    "section": "Capstone Project",
    "question": "I\u2019ve already submitted my project. Why can\u2019t I review any projects?",
    "answer": "Once the project submission deadline has passed, projects will be assigned to you for evaluation. You can't choose which projects to evaluate, and you can\u2019t review before the list has been released."
  },
  {
    "id": "9a2e2d2008",
    "course": "llm-zoomcamp",
    "section": "Capstone Project",
    "question": "How is my capstone project going to be evaluated?",
    "answer": "Each submitted project will be evaluated by three randomly assigned students who have also submitted the project.\n\nYou will also be responsible for grading the projects from three fellow students yourself. Please be aware that not complying with this rule implies you may fail to achieve the Certificate at the end of the course.\n\nThe final grade you receive will be the median score of the grades from the peer reviewers. The peer review criteria for evaluation must follow the guidelines defined here (TBA for link)."
  }
]</pre>
            </div>

            </details>



    -> Response received




            <details>
            <summary>Function call: <tt>search({"query":"assignments homework deadline slack t...)</tt></summary>
            <div>
                <b>Call</b>
                <pre>{"query":"assignments homework deadline slack telegram certificate exam quiz project submission FAQ llm-zoomcamp"}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>[
  {
    "id": "054f3fd58f",
    "course": "llm-zoomcamp",
    "section": "General Course-Related Questions",
    "question": "Where is the LLM Zoomcamp Telegram channel?",
    "answer": "The Telegram channel is [https://t.me/llm_zoomcamp](https://t.me/llm_zoomcamp).\n\nUse it for announcements. For technical discussion and questions, use the course Slack channel."
  },
  {
    "id": "930286278d",
    "course": "llm-zoomcamp",
    "section": "Capstone Project",
    "question": "Where can I find previous LLM Zoomcamp projects?",
    "answer": "You can browse previous LLM Zoomcamp project submissions here:\n\n- [2024 projects](https://courses.datatalks.club/llm-zoomcamp-2024/projects)\n- [2025 projects](https://courses.datatalks.club/llm-zoomcamp-2025/projects)\n\nThese pages show submitted repositories and can help you understand the expected scope and quality of capstone projects."
  },
  {
    "id": "20c5a1347e",
    "course": "llm-zoomcamp",
    "section": "General Course-Related Questions",
    "question": "Where can I track the LLM Zoomcamp syllabus, deadlines, homework, and progress?",
    "answer": "Use the [LLM Zoomcamp course management platform](https://courses.datatalks.club/llm-zoomcamp-2026/).\n\nIt contains the current cohort structure, homework, deadlines, and progress tracking. The process is the same as in other DataTalks.Club Zoomcamps."
  },
  {
    "id": "53f15299b6",
    "course": "llm-zoomcamp",
    "section": "Module 1 Homework",
    "question": "Where can I find the homework questions?",
    "answer": "Homework links are available in the course GitHub repo and in the course management platform.\n\nFor the 2026 Module 1 homework, use:\n\n- [Module 1 cohort materials](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/cohorts/2026/01-agentic-rag)\n- [Module 1 homework instructions](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/01-agentic-rag/homework.md)\n\nThe course platform is useful for submission and deadlines, but the GitHub homework instructions often contain important extra context."
  },
  {
    "id": "7d3a2e3505",
    "course": "llm-zoomcamp",
    "section": "Module 1 Homework",
    "question": "What should I do if homework questions feel unclear?",
    "answer": "First read the GitHub homework instructions, not only the course platform page.\n\nFor Module 1 in the 2026 cohort, start here:\n\n- [Module 1 cohort materials](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/cohorts/2026/01-agentic-rag)\n- [Module 1 homework instructions](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/01-agentic-rag/homework.md)\n\nThe homework follows the lesson workflow, but usually uses a different dataset or asks you to apply the same idea in a slightly different way.\n\nIf it is still unclear, ask in Slack and include the code or command output as text, not a screenshot. Follow the [asking questions guidelines](https://datatalks.club/docs/courses/zoomcamp-logistics/asking-questions/)."
  }
]</pre>
            </div>

            </details>



    -> Response received




            <div>
                <div><b>Assistant:</b></div>
                <div><p>I found course logistics info for <strong>llm-zoomcamp</strong>. Key points:</p>
<ul>
<li><strong>Live sessions / office hours:</strong> there are <strong>no separate live sessions for every module by default</strong>. Materials are pre-recorded, and live sessions are announced separately.</li>
<li><strong>Where to ask for help:</strong> use <strong>Slack</strong> for technical questions; announcements are also posted in the <strong>Telegram channel</strong>.</li>
<li><strong>Homework / deadlines / progress:</strong> use the <strong>course management platform</strong> for the current cohort’s syllabus, deadlines, homework, and progress tracking.</li>
<li><strong>Module 1 homework:</strong> the homework links and instructions are in the course GitHub repo, and the GitHub instructions may contain extra context beyond the platform page.</li>
<li><strong>Capstone project:</strong> it is <strong>individual</strong>, not a group project, and peer-reviewed after the submission deadline.</li>
</ul>
<p>If you want, I can help you find the exact FAQ entry for a specific course question. Are there other areas you want to explore?</p>
</div>
            </div>



    -> Response received




            <div>
                <div><b>Assistant:</b></div>
                <div><p>I’m here to help with course-related logistics or FAQ items. If you have a specific question about homework, deadlines, office hours, project submissions, or course channels, send it over and I’ll look it up. Are there other areas you want to explore?</p>
</div>
            </div>



    Chat ended.


[Back to TOC](#table-of-contents)

### Other Frameworks


As mentioned at the end of the section right before the ToyAIKit section, the agentic loop pipeline we set up follows essentially the same pattern--function calling, the agent loop, and tool definitions--as every other agent framework, whether it's LangChain, PydanticAI, or the OpenAI Agents SDK.  We now understand the basics of how any production framework works.  This module was framework agnostic, but it's worth knowing a little bit more about the frameworks mentioned above and how to install them.  

#### OpenAI Agents SDK
This is the official SDK from OpenAI for building agents. It uses the same Responses API we used throughout this module. It supports tool definition, multi-turn conversations, and handoffs between agents.  It's a good choice if you're already using OpenAI and want something official and well-maintained. To install using the ```uv``` package manager:

```uv add openai-agents```


#### PydanticAI
Alexey describes PydanticAI as a type-safe agent framework that supports multiple LLM providers.  (Type-safe agents are designed to strictly adhere to predefined schema or a set of data types.)  Tools are plain Python functions with type hints.  No wrappers are needed. Switching providers is as simple as changing the model string.

```uv add pydantic-ai```

This is Alexey's favorite.  While he appreciates the type-safety of PydanticAI, he says that other frameworks offer this as well.  The main reasons he likes it are the usability of PydanticAI and the team behind it.  He says it's a good choice if you want type safety and multi-provider support.

(I decided to look up some additional information about type-safe agents generally and PydanticAI more specifically.  A Google search summary turned up Mastra and DSPy, in addition to PydanticAI.  Looking at PydanticAI's website specifically, I learned that Pydantic Validation is the validation layer of the OpenAI SDK, the Google ADK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more.  As Pydantic says, "why use the derivative when you can go straight to the source?" Regarding type-safety, Pydantic states that PydanticAI is "...[designed] to give your IDE or AI coding agent as much context as possible for auto-completion and type checking, moving entire classes of errors from runtime to write-time for a bit of that Rust "if it compiles, it works" feel.")


#### LangChain / LangGraph
A popular framework with lots of integrations. LangChain handles the basics, and LangGraph adds graph-based workflows for more complex agent patterns.

Good choice if you need lots of integrations (vector stores, document loaders, etc.) and a large community.

#### Google ADK
The Agent Development Kit from Google. It exposes the same building blocks we've seen, like tools, instructions, and sessions. It also integrates with Google Cloud.  Best choice if you plan to use Gemini models and/or if your stack is on Google Cloud.

#### Others
Here are some other frameworks worth knowing about:

CrewAI - multi-agent orchestration
AutoGen - multi-agent conversations from Microsoft
Semantic Kernel - from Microsoft, supports C# and Python
Smolagents - lightweight agent framework from HuggingFace
Anthropic Tool Use - Anthropic's native tool use API

Pick one that fits your stack and your needs. The hard part is designing good tools and prompts - the loop is always the same.

[Back to TOC](#table-of-contents)

### A note about avoiding agents when a simpler tool will do the job


We just spent some time demonstrating a use case where agentic AI can be very useful.  That said, agents aren't always the best tool for the job, for a number of reasons:

* Cost:  There are likely to be more API calls per request because the loop can initiate many tool calls before the model is satisfied; furthermore, each iteration is another billed call that sends the full message history each time
* Time:  Because there may be many "round trips", each of which the model must complete before moving on to the next one, agentic approaches may involve a lot of lag time
* Monitoring/cognitive load:  Need to monitor cost, iteration count, and whether the agent is actually solving the problem or going in circles 
* Less predictable behavior:  LLMs are non-deterministic; the LLM can make different decisions on the same prompt run two different times, with different resulting paths

As with any problem-solving activity, the first step is figuring which tool is best for the job.  Many tasks can be accomplished by simpler approaches that are less costly (in terms of time, money, and monitoring / tracking results).  For example, before using an agentic approach, we should consider the following alternatives:

* Plain RAG--one search, one answer
* Parsing or templating a document into another form
* A single LLM call with no tools

If a simpler approach works well for your problem, use that.  Only if simpler approaches don't solve your problem should you reach for an agent loop.  Then you'll be more sure that you've selected the right tool for the job and are not needlessly wasting resources implementing it.

[Back to TOC](#table-of-contents)

