from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import time
import numpy as np
from typing import List, Dict

import os

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/models")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding model (QUERY ONLY)
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    cache_folder=MODEL_PATH
)

# Load existing ChromaDB
vectorstore = Chroma(
    collection_name="math_courses",
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {
            "section": {
                "$in": [
                    "definition",
                    "theorem",
                    "proof",
                    "note",
                    "remark",
                    "example"
                ]
            }
        }
    }
)

# Prompt (STRICT)
prompt = PromptTemplate.from_template("""
You are a university-level mathematics tutor.

Rules:
Use ONLY the information provided in the context.
Do NOT use external knowledge.
Do NOT introduce examples unless explicitly requested.
If the answer is not in the context, say:
  "The course does not provide this information."
Explain step 
by step with clear mathematical reasoning.

Context:
{context}

Question:
{question}

Answer:
""")

# LLM via Ollama
llm = ChatOllama(
    model="llama3.1",   
    temperature=0.1
)

# Build RAG CHAIN 
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ============= EVALUATION METRICS SECTION =============

class RAGMetrics:
    """Streamlined evaluation metrics for RAG pipeline (Precision, Recall, Response Time)"""
    
    def __init__(self):
        self.metrics_history = []
    
    @staticmethod
    def calculate_precision_at_k(relevant_docs: List[bool]) -> float:
        """Precision: % of retrieved docs that are relevant"""
        if not relevant_docs:
            return 0.0
        return sum(relevant_docs) / len(relevant_docs)
    
    @staticmethod
    def calculate_recall_at_k(relevant_docs: List[bool], total_relevant: int) -> float:
        """Recall: % of total relevant docs that were retrieved"""
        if total_relevant == 0:
            return 0.0
        return sum(relevant_docs) / total_relevant
    
    def evaluate_query(
        self, 
        query: str, 
        answer: str, 
        retrieval_time: float,
        generation_time: float, 
        retrieved_docs: List,
        expected_section: str = None,
        relevant_doc_ids: List[str] = None,
        total_relevant_docs: int = None
    ) -> Dict:
        """
        Calculate core metrics for a single query
        """
        
        # Basic metrics
        metrics = {
            "query": query,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
            "num_docs_retrieved": len(retrieved_docs),
            "answer_word_count": len(answer.split())
        }
        
        # ============= RETRIEVAL QUALITY METRICS =============
        
        # Determine relevance
        relevant_flags = []
        retrieved_sections = []
        
        for doc in retrieved_docs:
            doc_id = doc.metadata.get('chunk_id', '')
            section = doc.metadata.get('section', '')
            retrieved_sections.append(section)
            
            is_relevant = False
            if relevant_doc_ids is not None:
                is_relevant = str(doc_id) in [str(rid) for rid in relevant_doc_ids]
            elif expected_section is not None:
                is_relevant = section == expected_section
            else:
                is_relevant = True # Optimistic default
            
            relevant_flags.append(is_relevant)
        
        # Calculate Core Metrics
        metrics["precision"] = self.calculate_precision_at_k(relevant_flags)
        
        # Recall estimation
        if total_relevant_docs is not None:
            total_rel = total_relevant_docs
        elif expected_section is not None:
            total_rel = max(sum(relevant_flags), 1) # Estimate
        else:
            total_rel = sum(relevant_flags)
            
        metrics["recall"] = self.calculate_recall_at_k(relevant_flags, total_rel)
        
        # Hit Rate (any relevant found?)
        metrics["hit_rate"] = 1.0 if any(relevant_flags) else 0.0
        
        # Context Overlap (Grounding proxy)
        if retrieved_docs:
            context_text = " ".join([doc.page_content for doc in retrieved_docs])
            context_words = set(context_text.lower().split())
            answer_words = set(answer.lower().split())
            common_words = context_words.intersection(answer_words)
            metrics["context_overlap_ratio"] = len(common_words) / len(answer_words) if answer_words else 0
        else:
            metrics["context_overlap_ratio"] = 0
            
        metrics["sections_retrieved"] = retrieved_sections
        self.metrics_history.append(metrics)
        return metrics
    
    def print_metrics(self, metrics: Dict):
      
        print(f"Query: {metrics['query']}")
       
        
        print(f"\n⏱  PERFORMANCE:")
        print(f"   Retrieval:    {metrics['retrieval_time']*1000:.0f} ms")
        print(f"   Generation:   {metrics['generation_time']:.2f} s")
    
        print(f"\n RETRIEVAL ACCURACY:")
        print(f"   Precision:    {metrics['precision']:.1%} (Relevant/Retrieved)")
        print(f"   Recall:       {metrics['recall']:.1%} (Relevant/Total)")
        print(f"   Hit Rate:     {'✅' if metrics['hit_rate'] > 0 else '❌'}")
        
        print(f"\n CONTENT:")
        print(f"   Sections:     {', '.join(metrics['sections_retrieved'])}")
        print(f"   Overlap:      {metrics['context_overlap_ratio']:.1%} (Answer vs Context)")
        
        print(f"{'='*80}\n")
    
    def print_summary(self):
        """Print concise summary"""
        if not self.metrics_history:
            return
        
        print(f"\n{'='*80}")
        print(f"📈 AGGREGATE SUMMARY ({len(self.metrics_history)} Queries)")
        print(f"{'='*80}")
        
        # Averages
        avg_prec = np.mean([m['precision'] for m in self.metrics_history])
        avg_rec = np.mean([m['recall'] for m in self.metrics_history])
        avg_hit = np.mean([m['hit_rate'] for m in self.metrics_history])
        avg_gen_time = np.mean([m['generation_time'] for m in self.metrics_history])
        
        print(f" Avg Precision:   {avg_prec:.1%}")
        print(f" Avg Recall:      {avg_rec:.1%}")
        print(f" Hit Rate:        {avg_hit:.1%}")
        print(f"⏱ Avg Gen Time:    {avg_gen_time:.2f} s")


def query_with_metrics(
    query: str, 
    metrics_tracker: RAGMetrics,
    expected_section: str = None,
    relevant_doc_ids: List[str] = None,
    total_relevant_docs: int = None
):
    """
    Execute query and track comprehensive metrics
    """
    
    print(f"\n Query: {query}")
    if expected_section:
        print(f" Expected Section: {expected_section}")
    print("Processing...\n")
    
    # Measure retrieval time
    retrieval_start = time.time()
    retrieved_docs = retriever.invoke(query)
    retrieval_time = time.time() - retrieval_start
    
    # Measure generation time
    generation_start = time.time()
    answer = rag_chain.invoke(query)
    generation_time = time.time() - generation_start
    
    # Calculate and print metrics
    metrics = metrics_tracker.evaluate_query(
        query=query,
        answer=answer,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        retrieved_docs=retrieved_docs,
        expected_section=expected_section,
        relevant_doc_ids=relevant_doc_ids,
        total_relevant_docs=total_relevant_docs
    )
    
    print(f"✨ Answer:\n{answer}")
    metrics_tracker.print_metrics(metrics)
    
    return answer, metrics


# ============= TEST QUERIES WITH COMPREHENSIVE METRICS =============

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("🚀 RAG EVALUATION (SIMPLIFIED)")
    print(f"{'='*80}\n")
    
    # Initialize metrics tracker
    metrics_tracker = RAGMetrics()
        # Test Query 1 - Theorem
    
    query1 = "Explain the Pythagorean theorem and give me an example."
    answer1, metrics1 = query_with_metrics(
        query=query1,
        metrics_tracker=metrics_tracker,
        expected_section="theorem",
        total_relevant_docs=5
    )
    
    # Test Query 2 - Definition
    query2 = "What is a matrix? Define it clearly."
    answer2, metrics2 = query_with_metrics(
        query=query2,
        metrics_tracker=metrics_tracker,
        expected_section="definition",
        total_relevant_docs=3
    )
    
    # Test Query 3 - Proof
    query3 = "How do you prove the fundamental theorem of algebra?"
    answer3, metrics3 = query_with_metrics(
        query=query3,
        metrics_tracker=metrics_tracker,
        expected_section="proof",
        total_relevant_docs=2
    )
    
    # Test Query 4 - Example
    query4 = "Show me examples of solving quadratic equations."
    answer4, metrics4 = query_with_metrics(
        query=query4,
        metrics_tracker=metrics_tracker,
        expected_section="example",
        total_relevant_docs=4
    )
    
    # Print overall summary
    metrics_tracker.print_summary()
    
    # Export metrics to JSON for analysis
    import json
    from pathlib import Path
    
    output_file = Path("rag_metrics_detailed.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_tracker.metrics_history, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Metrics exported to: {output_file}")
 
