"""
Text embedder using pre-trained models for consistent vector dimensions.

This module provides fixed-dimension text embedding using Sentence Transformers
with fallback to basic methods when needed.
"""

import numpy as np
from typing import List, Union, Optional
import hashlib
import os


class Embedder:
    """Text embedder using Sentence Transformers for fixed dimensions"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_features: int = 384):
        """
        Initialize the embedder
        
        Args:
            model_name: Sentence Transformer model name
            max_features: Maximum vector dimension (will be ignored if model provides fixed dim)
        """
        self.model_name = model_name
        self.max_features = max_features
        self.method = "sentence_transformer"  # Add method attribute for compatibility
        self.model = None
        self._fitted = False
        self._vector_dimension = None
        
        # Try to import sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_transformers_available = True
        except ImportError:
            self._sentence_transformers_available = False
            print("Warning: sentence_transformers not available, falling back to basic methods")
    
    def _get_sentence_transformer(self):
        """Get the Sentence Transformer model"""
        if not self._sentence_transformers_available:
            raise ImportError("sentence_transformers not available. Install with: pip install sentence-transformers")
        
        from sentence_transformers import SentenceTransformer
        
        # Popular free models with fixed dimensions
        available_models = {
            "all-MiniLM-L6-v2": 384,      # Fast, good quality, 384 dimensions
            "all-MiniLM-L12-v2": 384,     # Better quality, 384 dimensions  
            "paraphrase-MiniLM-L6-v2": 384, # Good for similarity, 384 dimensions
            "multi-qa-MiniLM-L6-v2": 384,   # Good for Q&A, 384 dimensions
            "all-mpnet-base-v2": 768,     # High quality, 768 dimensions
            "all-distilroberta-v1": 768,  # Good balance, 768 dimensions
        }
        
        if self.model_name not in available_models:
            print(f"Warning: {self.model_name} not in recommended list, using anyway")
        
        try:
            model = SentenceTransformer(self.model_name)
            self._vector_dimension = model.get_sentence_embedding_dimension()
            return model
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to all-MiniLM-L6-v2")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._vector_dimension = model.get_sentence_embedding_dimension()
            return model
    
    def _get_fallback_embedder(self):
        """Get a fallback embedder when Sentence Transformers is not available"""
        try:
            # Import scikit-learn components directly for fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            class FallbackEmbedder:
                def __init__(self, max_features):
                    self.max_features = max_features
                    self.vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                    self._fitted = False
                
                def fit(self, texts):
                    self.vectorizer.fit(texts)
                    self._fitted = True
                
                def embed(self, text):
                    if not self._fitted:
                        self.fit([text])
                    vector = self.vectorizer.transform([text]).toarray()[0]
                    return vector.tolist()
                
                def embed_batch(self, texts):
                    if not self._fitted:
                        self.fit(texts)
                    vectors = self.vectorizer.transform(texts).toarray()
                    return [vector.tolist() for vector in vectors]
                
                def similarity(self, v1, v2):
                    v1_arr = np.array(v1).reshape(1, -1)
                    v2_arr = np.array(v2).reshape(1, -1)
                    return float(cosine_similarity(v1_arr, v2_arr)[0][0])
                
                def find_most_similar(self, query_vector, vectors, top_k=5):
                    similarities = []
                    for i, vector in enumerate(vectors):
                        sim = self.similarity(query_vector, vector)
                        similarities.append((i, sim))
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    return similarities[:top_k]
                
                def get_vector_dimension(self):
                    if hasattr(self.vectorizer, 'get_feature_names_out'):
                        return self.vectorizer.get_feature_names_out().shape[0]
                    return self.max_features
            
            return FallbackEmbedder(self.max_features)
            
        except ImportError:
            raise ImportError("No embedding method available. Install sentence-transformers or scikit-learn")
    
    def fit(self, texts: List[str] = None):
        """
        Initialize the model (for Sentence Transformers, this just loads the model)
        
        Args:
            texts: Not used for Sentence Transformers, kept for compatibility
        """
        if self._sentence_transformers_available:
            self.model = self._get_sentence_transformer()
            self._fitted = True
        else:
            # Fallback to basic embedder
            self.model = self._get_fallback_embedder()
            if texts:
                self.model.fit(texts)
            self._fitted = True
    
    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string
        
        Args:
            text: Text string to embed
            
        Returns:
            List of float values representing the text vector
        """
        if not self._fitted:
            self.fit()
        
        if self._sentence_transformers_available and self.model:
            # Use Sentence Transformers
            vector = self.model.encode(text)
            return vector.tolist()
        else:
            # Fallback to basic embedder
            return self.model.embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of vectors, each vector is a list of floats
        """
        if not self._fitted:
            self.fit()
        
        if self._sentence_transformers_available and self.model:
            # Use Sentence Transformers batch processing
            vectors = self.model.encode(texts)
            return [vector.tolist() for vector in vectors]
        else:
            # Fallback to basic embedder
            return self.model.embed_batch(texts)
    
    def similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate cosine similarity
        similarity = np.dot(v1_norm, v2_norm)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def find_most_similar(self, query_vector: List[float], vectors: List[List[float]], top_k: int = 5) -> List[tuple]:
        """
        Find the most similar vectors to a query vector
        
        Args:
            query_vector: Query vector to compare against
            vectors: List of vectors to compare with
            top_k: Number of top similar vectors to return
            
        Returns:
            List of tuples (index, similarity_score) sorted by similarity
        """
        similarities = []
        for i, vector in enumerate(vectors):
            sim = self.similarity(query_vector, vector)
            similarities.append((i, sim))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of vectors produced by this embedder"""
        if self._vector_dimension:
            return self._vector_dimension
        elif self.model and hasattr(self.model, 'get_sentence_embedding_dimension'):
            return self.model.get_sentence_embedding_dimension()
        elif hasattr(self.model, 'get_vector_dimension'):
            return self.model.get_vector_dimension()
        else:
            return self.max_features
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        info = {
            "model_name": self.model_name,
            "vector_dimension": self.get_vector_dimension(),
            "method": "sentence_transformers" if self._sentence_transformers_available else "fallback",
            "fitted": self._fitted
        }
        
        if self._sentence_transformers_available and self.model:
            info["model_type"] = type(self.model).__name__
            info["available_models"] = [
                "all-MiniLM-L6-v2 (384d)",
                "all-MiniLM-L12-v2 (384d)", 
                "paraphrase-MiniLM-L6-v2 (384d)",
                "multi-qa-MiniLM-L6-v2 (384d)",
                "all-mpnet-base-v2 (768d)",
                "all-distilroberta-v1 (768d)"
            ]
        
        return info


# Convenience function to get recommended models
def get_recommended_models():
    """Get list of recommended Sentence Transformer models"""
    return {
        "fast": "all-MiniLM-L6-v2",      # 384d, fastest
        "balanced": "all-MiniLM-L12-v2",  # 384d, good balance
        "similarity": "paraphrase-MiniLM-L6-v2",  # 384d, best for similarity
        "qa": "multi-qa-MiniLM-L6-v2",   # 384d, best for Q&A
        "high_quality": "all-mpnet-base-v2",      # 768d, highest quality
        "robust": "all-distilroberta-v1"  # 768d, good robustness
    }
