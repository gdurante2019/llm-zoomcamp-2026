INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class CustomRAGBase(RAGBase):
    def search(self, query, num_results=5):
        # Use your custom index search
        return self.index.search(query, num_results=num_results)
    
    def build_context(self, search_results):
        # Build context from 'filename' and 'content'
        lines = []
        for doc in search_results:
            lines.append(f"File: {doc['filename']}")
            lines.append(doc['content'])
            lines.append('')
        return '\n'.join(lines).strip()
    
    def llm(self, prompt):
        # Return the FULL response, not just text
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages
        )
        return response  # Return full response!
    
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response  # Now returns response with .usage

assistant = CustomRAGBase(index=index, llm_client=openai_client)
response = assistant.rag("How does the agentic loop keep calling the model until it stops?")
answer = response.choices[0].message.content
print(answer)

# Now you can calculate cost
cost = (
    response.usage.prompt_tokens * input_price +
    response.usage.completion_tokens * output_price
)
print(f"Cost: ${cost}")
