# Module 3: AI Orchestration with Kestra


## Introduction
_(Text in this section is verbatim from the lesson)_

In this module, we learn how to orchestrate AI workflows using Kestra, an open-source orchestration platform. We start from the context problem that makes generic AI assistants unreliable, and end with autonomous multi-agent systems that can research, reason, and act without a fixed predetermined sequence of steps.

This module is created and taught by Will Russell from Kestra. Big thanks to Will and the Kestra team for making it possible.

### Part 1: Using AI in Workflows
The first three lessons cover the theory behind using AI reliably in workflows — why context matters, why generic AI assistants fail, and how to set up your environment.

1. Introduction - Learning objectives, prerequisites, and why AI for workflows
2. Context Engineering - Why generic AI assistants produce incorrect Kestra flows
3. Setting up Kestra - Kestra setup, API keys, and importing the example flows

### Part 2: AI Copilot
Part 2 shows how to use Kestra's AI Copilot to generate and refine flows by describing your inputs and goal, rather than building each step manually.

4. AI Copilot - Generating and refining flows with Kestra's built-in AI Copilot

### Part 3: RAG Workflows
Part 3 shows how to ground AI responses in real data using Retrieval Augmented Generation, so the model works from facts instead of guessing. For a deeper dive into RAG and vector search, see Module 2.

5. Retrieval Augmented Generation - Ingesting documents, creating embeddings, and querying with context


### Part 4: Agentic Workflows
Part 4 introduces AI agents that make autonomous decisions, use tools, and collaborate in multi-agent systems to complete complex tasks.

6. AI Agents - Autonomous task execution, available tools, and observability
7. Multi-Agent Systems - Specialized agents collaborating on complex tasks

### Part 5: Best Practices
Part 5 covers what you need to know before going to production - cost, security, observability, and when to use each approach.

8. Best Practices - Cost, security, observability, and production readiness
9. Next Steps - Resources, further reading, and where to go from here

## AI Orchestration

For this module, we gain experience with an orchestration platform.  Students can use whatever orchestration platform they wish to.  Because Kestra is sponsoring the course and the lessons in Module 3 are geared around Kestra, that's what I'll be using.  

### Setting up Kestra

The first step is getting Kestra running locally.  We'll be using a Docker container for this.  

#### Step 1: Start Kestra
This module includes a ```docker-compose.yml``` with Kestra pre-configured.  Because I'm running everything locally rather than through GitHub Codespaces, I first need to start up the Docker Desktop app, then launch Docker compose in the terminal: 

```docker compose up -d```

Once the container is launched, then we can access Kestra at http://localhost:8080/.


## Homework

The homework instructions are included as a markdown file in this folder.  The output of the workflow executions I ran in Kestra is available in this folder as an Excel file. 

