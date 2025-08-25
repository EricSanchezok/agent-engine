"""
Advanced Memory Usage Example

This demonstrates how to use the Memory class with Sentence Transformers
for fixed-dimension, high-quality embeddings.
"""

from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_engine.memory import Memory, get_recommended_models


def main():
    """Demonstrate advanced memory usage"""
    print("Advanced Memory Usage Example\n")
    
    # Show available models
    print("Available Sentence Transformer models:")
    models = get_recommended_models()
    for category, model_name in models.items():
        print(f"  {category:12}: {model_name}")
    
    print("\n" + "="*60)
    
    # Create memory with sentence transformer (default: all-MiniLM-L6-v2)
    print("1. Creating memory with Sentence Transformers...")
    memory = Memory("advanced_example")
    print(f"   Memory info: {memory.get_info()}")
    print(f"   Vector dimension: {memory.embedder.get_vector_dimension()}")
    
    # Add content about different topics
    print("\n2. Adding content to memory...")
    
    content_items = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "metadata": {"topic": "AI/ML", "difficulty": "intermediate", "type": "definition"}
        },
        {
            "text": "Neural networks are computational models inspired by biological neurons, used in deep learning for pattern recognition.",
            "metadata": {"topic": "Deep Learning", "difficulty": "advanced", "type": "concept"}
        },
        {
            "text": "Python is a high-level programming language known for its simplicity and readability, widely used in data science.",
            "metadata": {"topic": "Programming", "difficulty": "beginner", "type": "language"}
        },
        {
            "text": "Natural language processing combines linguistics, computer science, and AI to enable computers to understand human language.",
            "metadata": {"topic": "NLP", "difficulty": "advanced", "type": "field"}
        },
        {
            "text": "Data science involves extracting insights from data using statistical analysis, machine learning, and visualization techniques.",
            "metadata": {"topic": "Data Science", "difficulty": "intermediate", "type": "field"}
        }
    ]
    
    for item in content_items:
        memory.add(item["text"], metadata=item["metadata"])
        print(f"   Added: {item['text'][:50]}...")
    
    print(f"\n   Total items: {memory.count()}")
    
    # Test semantic search
    print("\n3. Testing semantic search...")
    
    search_queries = [
        "artificial intelligence and learning",
        "computer programming languages",
        "understanding human language",
        "data analysis and visualization",
        "biological neural systems"
    ]
    
    for query in search_queries:
        print(f"\n   Query: '{query}'")
        results = memory.search(query, top_k=2)
        for i, (content, score, metadata) in enumerate(results):
            print(f"     {i+1}. Score: {score:.4f}")
            print(f"        Content: {content[:60]}...")
            print(f"        Topic: {metadata.get('topic', 'N/A')}")
    
    # Test similarity between specific content
    print("\n4. Testing content similarity...")
    
    # Get all content
    all_contents = memory.get_all_contents()
    
    # Compare first two items
    if len(all_contents) >= 2:
        result1 = memory.get_by_content(all_contents[0])
        result2 = memory.get_by_content(all_contents[1])
        
        if result1 and result2:
            vector1, metadata1 = result1
            vector2, metadata2 = result2
            
            similarity = memory.embedder.similarity(vector1, vector2)
            print(f"   Similarity between '{all_contents[0][:30]}...' and '{all_contents[1][:30]}...': {similarity:.4f}")
    
    # Test metadata filtering (conceptual - you could extend Memory class for this)
    print("\n5. Content analysis by topic...")
    
    all_metadata = memory.get_all_metadata()
    topic_counts = {}
    
    for metadata in all_metadata:
        topic = metadata.get('topic', 'Unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print("   Content distribution by topic:")
    for topic, count in topic_counts.items():
        print(f"     {topic}: {count} items")
    
    print("\n" + "="*60)
    print("Advanced Memory Example Completed Successfully! 🎉")
    print("\nKey Benefits:")
    print("✅ Fixed vector dimensions (384 for all-MiniLM-L6-v2)")
    print("✅ Better semantic understanding than TF-IDF")
    print("✅ Consistent similarity scores")
    print("✅ Professional-grade embeddings")
    print("✅ No external API calls needed")


if __name__ == "__main__":
    main()
