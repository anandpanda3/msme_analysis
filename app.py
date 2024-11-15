import streamlit as st
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from rank_bm25 import BM25Okapi
import logging
from tqdm import tqdm
import asyncio
import json
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Region definitions
REGIONS = {
    "North India": "Hindi with English translation",
    "South India (Tamil Nadu)": "Tamil with English translation",
    "South India (Kerala)": "Malayalam with English translation",
    "South India (Karnataka)": "Kannada with English translation",
    "South India (Andhra/Telangana)": "Telugu with English translation",
    "West India (Maharashtra)": "Marathi with English translation",
    "West India (Gujarat)": "Gujarati with English translation",
    "East India (West Bengal)": "Bengali with English translation",
    "All India": "English"
}

class DocumentProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        
    def preprocess(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(self.tokenizer.decode(chunk, skip_special_tokens=True))
        
        return chunks

class MultilingualRAGSystem:
    def __init__(self, collection_name: str = "msme_embeddings"):
        load_dotenv()
        
        # Initialize clients
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.collection_name = collection_name
        
        # Initialize models
        logger.info("Loading embedding model...")
        self.encoder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
        logger.info("Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.document_processor = DocumentProcessor()
        
        # Initialize storage
        self.bm25 = None
        self.documents = []
        self.document_lookup = {}
        self.conversation_history = []
        
        # Create Qdrant collection if it doesn't exist
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with proper configuration"""
        try:
            collections = self.qdrant_client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,  # MPNet base embedding size
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            logger.info("Generating embeddings...")
            embeddings = []
            batch_size = 32
            
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.encoder.encode(batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def load_data_from_csv(self, csv_path: str, regenerate_embeddings: bool = False):
        """Load and process data from CSV file"""
        logger.info("Loading data from CSV...")
        df = pd.read_csv(csv_path)
        
        processed_documents = []
        raw_texts = []
        
        # Process documents
        for idx, row in tqdm(df.iterrows(), desc="Processing documents"):
            text = row['msme']
            processed_documents.append({
                "id": str(idx),
                "text": text,
                "metadata": {
                    "state": row.get("state", ""),
                    "district": row.get("district", ""),
                    "language": row.get("language", "")
                }
            })
            raw_texts.append(text)
            self.document_lookup[str(idx)] = text
        
        # Initialize BM25
        tokenized_corpus = [doc["text"].lower().split() for doc in processed_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.documents = processed_documents
        
        # Generate or load embeddings
        if regenerate_embeddings or 'embedding' not in df.columns:
            logger.info("Generating new embeddings...")
            embeddings = self.generate_embeddings(raw_texts)
            df['embedding'] = embeddings.tolist()
            # Save embeddings back to CSV
            df.to_csv(csv_path, index=False)
        else:
            logger.info("Loading existing embeddings...")
            df['embedding'] = df['embedding'].apply(eval)
        
        # Upload to Qdrant
        batch_size = 100
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
        
        for i in tqdm(range(0, len(df), batch_size), desc="Uploading to Qdrant", total=total_batches):
            batch_df = df.iloc[i:i + batch_size]
            points = [
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "text": row["msme"],
                        "state": row.get("state", ""),
                        "district": row.get("district", ""),
                        "language": row.get("language", "")
                    }
                )
                for idx, (embedding, row) in enumerate(zip(
                    batch_df['embedding'],
                    batch_df.to_dict(orient="records")
                ), start=i)
            ]
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        logger.info(f"Indexed {len(processed_documents)} documents")

    def _get_conversation_context(self) -> str:
        """Get recent conversation history as context"""
        return "\n".join([
            f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
            for i, msg in enumerate(self.conversation_history[-4:])
        ])

    async def _get_llm_response(self, prompt: str, context: str = "", region: str = "All India") -> str:
        """Get response from Groq LLM in specified language"""
        try:
            system_prompt = f"""You are a friendly and helpful AI assistant specializing in MSME policies. 
            Your responses should be conversational and engaging while being informative. 
            When someone asks general questions or greetings, respond naturally before guiding them to MSME-related topics.
            
            IMPORTANT: For this conversation, give your response in {REGIONS[region]}. 
            If the region is not "All India", provide the response in the regional language followed by its English translation in parentheses.
            
            Base your response primarily on the provided context, but you can add general knowledge about MSMEs when appropriate.
            Always maintain accuracy and provide specific details from the context when available."""

            conversation_context = self._get_conversation_context()
            
            # Combine context and prompt
            full_prompt = f"""Previous conversation:\n{conversation_context}\n
            Context information:\n{context}\n\nUser's question: {prompt}
            
            Please provide a helpful response based on the context and question."""

            chat_completion = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise

    def _hybrid_search(self, question: str, top_k: int = 5) -> List[Dict]:
        try:
            # Get BM25 results
            tokenized_query = question.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[-top_k:]
            bm25_results = [{
                "id": self.documents[i]["id"],
                "text": self.documents[i]["text"],
                "metadata": self.documents[i]["metadata"],
                "score": float(bm25_scores[i])  # Convert numpy float to Python float
            } for i in top_bm25_indices]
            
            # Get vector search results
            query_embedding = self.encoder.encode(question)
            vector_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            # Convert vector results to same format as BM25 results
            vector_docs = [{
                "id": str(hit.id),
                "text": hit.payload["text"],
                "metadata": {
                    "state": hit.payload.get("state", ""),
                    "district": hit.payload.get("district", ""),
                    "language": hit.payload.get("language", "")
                },
                "score": float(hit.score)  # Ensure score is a Python float
            } for hit in vector_results]
            
            # Combine and normalize scores
            all_results = bm25_results + vector_docs
            
            # Normalize scores between 0 and 1 for each method
            if all_results:
                max_score = max(r["score"] for r in all_results)
                min_score = min(r["score"] for r in all_results)
                score_range = max_score - min_score if max_score != min_score else 1
                
                for result in all_results:
                    result["score"] = (result["score"] - min_score) / score_range
            
            # Deduplicate results by keeping highest scoring version
            unique_results = {}
            for result in all_results:
                doc_id = result["id"]
                if doc_id not in unique_results or result["score"] > unique_results[doc_id]["score"]:
                    unique_results[doc_id] = result
            
            # Sort by score and take top_k
            final_results = sorted(
                unique_results.values(),
                key=lambda x: x["score"],
                reverse=True
            )[:top_k]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def _should_use_rag(self, query: str) -> bool:
        """Determine if query needs RAG or is conversational"""
        general_phrases = [
            "hello", "hi", "hey", "how are you", "good morning",
            "good afternoon", "good evening", "thanks", "thank you",
            "bye", "goodbye"
        ]
        return not any(phrase in query.lower() for phrase in general_phrases)

    async def query(self, question: str, region: str, top_k: int = 5) -> Dict:
        """Handle user query with hybrid search and reranking"""
        try:
            self.conversation_history.append(question)
            
            # Handle conversational queries
            if not self._should_use_rag(question):
                response = await self._get_llm_response(question, region=region)
                self.conversation_history.append(response)
                return {
                    "answer": response,
                    "conversation_type": "general",
                    "supporting_documents": []
                }
            
            # Perform hybrid search
            search_results = self._hybrid_search(question, top_k)
            
            if search_results:
                # Rerank results using cross-encoder
                passages = [r["text"] for r in search_results]
                rerank_scores = self.cross_encoder.predict([
                    (question, passage) for passage in passages
                ])
                
                # Sort by reranking scores
                reranked_results = [x for _, x in sorted(
                    zip(rerank_scores, passages),
                    reverse=True
                )][:top_k]
            else:
                reranked_results = []
            
            # Generate response
            context = "\n".join(reranked_results) if reranked_results else ""
            response = await self._get_llm_response(question, context, region)
            
            self.conversation_history.append(response)
            
            return {
                "answer": response,
                "conversation_type": "msme",
                "supporting_documents": reranked_results
            }
            
        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}")
            raise

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = "All India"

def load_rag_system():
    """Initialize and load the RAG system"""
    try:
        with st.spinner("Loading MSME data and initializing models... This may take a few minutes..."):
            rag_system = MultilingualRAGSystem()
            rag_system.load_data_from_csv('msme_data_with_embeddings.csv')
            st.session_state.rag_system = rag_system
            st.session_state.data_loaded = True
            st.success("System initialized successfully!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")

async def process_query(query: str, region: str) -> Dict:
    """Process user query and return response"""
    if not st.session_state.data_loaded:
        raise Exception("System not initialized. Please wait for data loading to complete.")
    return await st.session_state.rag_system.query(query, region)

def main():
    st.set_page_config(page_title="MSME Policy Assistant", page_icon="🤖", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for region selection
    with st.sidebar:
        st.title("🌍 Region Settings")
        selected_region = st.selectbox(
            "Select your region",
            options=list(REGIONS.keys()),
            index=list(REGIONS.keys()).index(st.session_state.selected_region)
        )
        
        # Add regenerate embeddings option
        if st.button("Regenerate Embeddings"):
            if st.session_state.rag_system:
                with st.spinner("Regenerating embeddings..."):
                    st.session_state.rag_system.load_data_from_csv('msme_data_with_embeddings.csv', regenerate_embeddings=True)
                st.success("Embeddings regenerated successfully!")
        
        if selected_region != st.session_state.selected_region:
            st.session_state.selected_region = selected_region
            st.session_state.messages = []  # Clear chat history when region changes
    
    # Main content
    st.title("👋 Welcome to MSME Policy Assistant")
    st.markdown(f"""
    I'm here to help you with any questions about MSME policies! Feel free to ask anything or just say hello.
    Currently set to respond in: **{REGIONS[st.session_state.selected_region]}**
    """)
    
    # Initialize RAG system if not already done
    if not st.session_state.data_loaded:
        load_rag_system()
    
    # Only show chat interface if system is loaded
    if st.session_state.data_loaded:
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "supporting_docs" in message and message["supporting_docs"]:
                        with st.expander("View Supporting Documents"):
                            st.markdown("### Reference Documents")
                            for i, doc in enumerate(message["supporting_docs"], 1):
                                st.markdown(f"**Document {i}:**")
                                st.markdown(doc)
        
        # User input
        if prompt := st.chat_input("Ask me anything about MSMEs or just say hi! 😊"):
            # Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking... 🤔"):
                        response = asyncio.run(process_query(prompt, st.session_state.selected_region))
                        
                        st.markdown(response["answer"])
                        
                        if response["conversation_type"] == "msme" and response["supporting_documents"]:
                            with st.expander("View Supporting Documents"):
                                st.markdown("### Reference Documents")
                                for i, doc in enumerate(response["supporting_documents"], 1):
                                    st.markdown(f"**Document {i}:**")
                                    st.markdown(doc)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "supporting_docs": response["supporting_documents"]
                        })
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()