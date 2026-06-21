# Part 1:  Building the foundation for our RAG agent 

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

## What is RAG and why do we need it?

### Plain LLMs lack our data
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

### Adding context manually

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


### Retrieval plus generation

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

## The Course FAQ Dataset

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

### Using this data

In the RAG pipeline, this dataset is our knowledge base:

* We index all the documents (the search step)
* When a student asks a question, we search the index
* The search returns the most relevant FAQ entries
* We give those entries to the LLM as context
* The LLM generates an answer based on the context

The question and answer fields contain the text we'll search through. The course field lets us filter by course. For example, if a student asks about the data engineering course, we skip results from the ML course. The section field helps with ranking - knowing which part of the course a question belongs to is useful context.


### A note on data preparation

In our case, the data is already prepared. Alexey maintains this FAQ website and made sure the data comes back in a convenient JSON format. So we don't need to do much to get it ready. He cleaned a lot of this data with the help of an LLM (a handy use case on its own).

In reality, data preparation is often the most time-consuming part of building a RAG system. You may need to scrape websites, parse PDFs, and clean and chunk documents. That work isn't visible here, but he did plenty of it ahead of time.

We keep the focus on the GenAI side in this course. For our projects, we should expect to spend significant time on data preparation before we get to this point.

In the next section, we'll build the search index.

## Search

### Search basics

At its core, every search engine does the same thing. It takes a query, scores every document for similarity, and returns the top results.

For each document in the database, you compute this score. Then you rank all documents by score and return the top N. What makes a search engine different from another search engine is what sim actually computes.

* Text/lexical search (covered in this section): sim counts how many words the query and the document share. It looks at the surface form, the actual words, and matches them exactly.
* Vector/semantic search (covered in module 2): sim compares the meaning of the query and the document. Same function, different similarity measure.

Consider these two questions:

* "Can I still join the course after the start date?"
* "Is it possible to enroll late?"

They mean the same thing, but share almost no keywords. "Join" is not "enroll", "course" is absent, "start date" is not "late". A text search engine would struggle to match them, because it only sees words.

We'll see how vector search solves this later. For now, let's build text search with minsearch.

### Indexing with minsearch

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

### Trying a search

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


### Boosting fields

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

### Filtering by course

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



### Wrapping it in a function

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



## Building the Prompt

The LLM doesn't see our documents unless we pass them in. So we need to build a prompt that includes the user's question and the search results.

When we build AI systems, we usually split the prompt into two parts:

* Instructions (also called the system prompt): this tells the LLM how to behave. It never changes, so it's the same for every request.
* User prompt: this changes with every request. It carries the actual question and the retrieved context.

We split them because the instructions are fixed and the user prompt is not. Keeping them apart makes the fixed part easy to reuse and the changing part easy to build fresh each time.

### Instructions

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

### The user prompt template

The user prompt template has placeholders for the question and the context:


```python
USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""
```

### Building the context

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

### Building the prompt

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

## The LLM

The last component of our RAG pipeline is the LLM. It takes the prompt we built and generates an answer.

### Sending the prompt to the LLM

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

### Exploring the response

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



### Calculating the price

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

### Message history

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

### The LLM function

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

### Full RAG

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

### Try more questions

Try a few more:


```python
rag("How do I get a certificate?")
```




    'You can get a certificate only if you finish the course with a **live cohort**. Certificates are **not** awarded for the self-paced mode.\n\nTo be eligible, you need to **complete and pass the Capstone project**. You also need to **peer-review 3 capstones** during the course run, since peer review is required while the course is active.\n\nIf you want your real name on the certificate, update the **official name** field in your course profile.'



Notice how the answers reference specific courses and sections. The LLM reads from our knowledge base before answering; that's how RAG works.

This approach is modular. You can swap the search backend, the prompt template, or the LLM model. Nothing else needs to change. Later when we replace minsearch with sqlitesearch, only the ```search``` function changes.

## RAG Helper

In the previous lessons, we built the RAG flow piece by piece - search, then the prompt, then the LLM call. The pipeline works, but every time we want to use it, we need to repeat the same code.

We'll use this code throughout the course, so let's put it into two reusable files:

* ingest.py - loading data and building the search index
* rag_helper.py - the RAG logic (search, prompt, LLM)

Then in notebooks, we just import from these files and use them.

### ingest.py

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

### Using it in a notebook

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



## Part 2:  Persistent RAG

Up until this point, we've been using ```minsearch```, which is fine if our database is relatively small because the indexing is fast.  ```minsearch``` is in-memory; it is a bunch of python libraries bound to the process in which it is running.  Once you stop the process, the data disappears.  When you re-start the process, the indexing has to happen all over again.  This breaks down as the database grows, needlessly consuming time and resources.  

The solution to this is to separate the ingestion part of the process from querying.  One process writes the data to a persistent search index, while another process reads from it.  These two processes run independently, only sharing the index between them.   

There are several persistent search backend for this, such as Elasticsearch, OpenSearch, Qdrant, and ```sqlitesearch```.  In this module, we use ```sqlitesearch```, a library Alexey wrote.  It is a lightweight search library and has the same API as ```minsearch```, so we can easily drop it in to our code.  It leverages SQLite, which already ships with python, and puts an easier-to-use wrapper around python's full-text search engine.  

## Demonstrating Persistent RAG in Action:  Exercise

Alexey instructs us to create two separate notebooks to demonstrate how the persistence process works.  One notebook ingests and indexes the document, creating an indexed database.  The other runs queries against that database.  That's how the two processes connect to each other.  

### Ingestion Notebook

See https://github.com/gdurante2019/llm-zoomcamp-2026/blob/main/01-agentic-rag/sqlite-ingest.ipynb for the ingestion notebook.

### Query notebook

See https://github.com/gdurante2019/llm-zoomcamp-2026/blob/main/01-agentic-rag/query-notebook.ipynb for the notebook executing the query.
