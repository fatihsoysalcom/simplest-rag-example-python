# This script demonstrates the core concept of Retrieval Augmented Generation (RAG).
# It uses a simple in-memory knowledge base, a basic keyword retrieval mechanism,
# and a mock LLM to show how external information can improve response accuracy.

# 1. Knowledge Base: A collection of documents (strings) that our RAG system can query.
# In a real RAG system, this would be a vector database or a searchable document store.
knowledge_base = [
    "Acme Corp was founded in 1990 by John Doe.",
    "The main product of Acme Corp is the 'SuperWidget 3000', known for its efficiency and durability.",
    "SuperWidget 3000 was launched in 2022 and has received positive reviews for its innovative features.",
    "Acme Corp also offers consulting services in AI and data analytics to enterprise clients.",
    "The company's headquarters are located in Tech City, a hub for technological innovation."
]

# 2. Retrieval Function: Simulates finding relevant documents based on a query.
# This is a very basic keyword-matching retrieval. Real systems use embeddings for semantic search.
def retrieve_documents(query, kb):
    relevant_docs = []
    query_lower = query.lower()
    # Split query into words and check if any word appears in a document
    # This is a simplified approach for demonstration purposes.
    query_keywords = set(word for word in query_lower.split() if len(word) > 2) # Ignore very short words

    for doc in kb:
        doc_lower = doc.lower()
        if any(keyword in doc_lower for keyword in query_keywords):
            relevant_docs.append(doc)
    return relevant_docs

# 3. Mock LLM Function: Simulates a Large Language Model's response.
# This mock LLM is designed to give a generic/incorrect answer without context,
# but a specific/correct answer when relevant context is provided in the prompt.
def mock_llm_generate(prompt):
    # Simulate a 'naive' LLM that might hallucinate or be unspecific without context
    if "main product of Acme Corp" in prompt and "Context:" not in prompt:
        return "I'm not sure about Acme Corp's main product, but many companies produce various gadgets."
    elif "founded" in prompt and "Acme Corp" in prompt and "Context:" not in prompt:
        return "Companies are founded at different times, I don't have specific details for Acme Corp."

    # Simulate an LLM that uses provided context to answer accurately
    if "SuperWidget 3000" in prompt and "main product" in prompt and "Context:" in prompt:
        return "Based on the provided information, the main product of Acme Corp is the 'SuperWidget 3000', known for its efficiency and durability."
    elif "Acme Corp" in prompt and "founded" in prompt and "Context:" in prompt:
        return "According to the context, Acme Corp was founded in 1990 by John Doe."
    elif "consulting services" in prompt and "Acme Corp" in prompt and "Context:" in prompt:
        return "The provided context states that Acme Corp offers consulting services in AI and data analytics."
    elif "headquarters" in prompt and "Acme Corp" in prompt and "Context:" in prompt:
        return "The information indicates that Acme Corp's headquarters are located in Tech City."
    elif "launch year of SuperWidget 3000" in prompt and "Context:" in prompt:
        return "The SuperWidget 3000 was launched in 2022."
    else:
        return "I'm sorry, I cannot provide a specific answer based on the given information."

# 4. RAG Orchestration: Combines retrieval and generation.
def simple_rag_system(query, kb, llm_model):
    print(f"\n--- Processing Query: '{query}' ---")

    # Step 1: Retrieval - Find relevant documents from the knowledge base.
    retrieved_docs = retrieve_documents(query, kb)
    print(f"Retrieved {len(retrieved_docs)} document(s).")

    # Step 2: Augmentation - Create an augmented prompt with the retrieved context.
    if retrieved_docs:
        context = "\n".join(retrieved_docs)
        augmented_prompt = (
            f"Based on the following context, please answer the question accurately.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
    else:
        # If no documents are retrieved, the prompt is not augmented.
        augmented_prompt = f"Question: {query}\nAnswer:"
        print("No relevant documents found. Proceeding with unaugmented prompt.")

    # Step 3: Generation - Send the augmented (or unaugmented) prompt to the LLM.
    response = llm_model(augmented_prompt)
    return response

# --- Demonstration --- 
if __name__ == "__main__":
    queries = [
        "What is the main product of Acme Corp?",
        "When was Acme Corp founded?",
        "What consulting services does Acme Corp offer?",
        "Where are Acme Corp's headquarters?",
        "Tell me about the launch year of SuperWidget 3000."
    ]

    print("\n### Demonstrating Naive LLM (without RAG) ###")
    for q in queries:
        naive_prompt = f"Question: {q}\nAnswer:"
        naive_response = mock_llm_generate(naive_prompt)
        print(f"\nQuery: {q}")
        print(f"Naive LLM Response: {naive_response}")

    print("\n\n### Demonstrating Simple RAG System ###")
    for q in queries:
        rag_response = simple_rag_system(q, knowledge_base, mock_llm_generate)
        print(f"RAG LLM Response: {rag_response}")
