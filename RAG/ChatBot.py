#!/usr/bin/env python3
# swedish_osm_rag_pipeline.py - RAG pipeline for Swedish OSM data using Gemini

import os
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
import chromadb
from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from OSM import SwedishOSMIndexer


class OSMRAGState(TypedDict):
    """State for the OSM RAG application"""
    question: str
    context: List[Document]
    answer: str
    location_filter: Optional[str]
    coordinates: Optional[Dict[str, float]]


class SwedishOSMRAG:
    """RAG system for Swedish OpenStreetMap data using Gemini"""
    
    def __init__(self, 
                 db_path: str = "./sweden_osm_rag",
                 google_api_key = "ENTER-API-KEY-HERE",
                 model_name: str = "gemini-2.5-flash"):
        """
        Initialize the RAG system
        
        Args:
            db_path: Path to ChromaDB database
            google_api_key: Google API key for Gemini (or set GOOGLE_API_KEY env var)
            model_name: Gemini model to use
        """
        self.indexer = SwedishOSMIndexer(db_path)
        
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model= "gemini-2.5-flash",
            temperature=0.1,
            max_tokens=1000,
            google_api_key = google_api_key
        )
        
        # Create prompt template for Swedish geography
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant for Swedish geography and locations based on OpenStreetMap data. 
            
Your role is to:
- Answer questions about Swedish places, cities, towns, natural features, and infrastructure
- Provide accurate coordinates in EPSG:3006 (SWEREF99 TM) coordinate system when available
- Give detailed, helpful information about locations, including population, elevation, and other attributes
- Help with navigation, tourism, and geographic information queries
- Explain Swedish place names and their characteristics

Use the provided context from OpenStreetMap data to answer questions accurately. If coordinates are mentioned, they are in EPSG:3006 format (Swedish national grid).

Context from Swedish OpenStreetMap data:
{context}

Remember:
- EPSG:3006 coordinates: X represents easting (longitude-like), Y represents northing (latitude-like)
- Provide practical information that would be useful for someone visiting or learning about Sweden
- If you're not certain about something from the context, say so clearly"""),
            ("user", "{question}")
        ])
        
        # Build the graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph pipeline"""
        graph_builder = StateGraph(OSMRAGState)
        
        # Add nodes
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)
        
        # Add edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        # Compile graph
        self.graph = graph_builder.compile()
    
    def _retrieve(self, state: OSMRAGState) -> Dict[str, Any]:
        """Retrieve relevant documents from OSM data"""
        question = state["question"]
        # location_filter = state.get("location_filter")
        
        n_results = 3  # Get more results for better context
        category = None
        
        question_lower = question.lower()
        if any(word in question_lower for word in ['city', 'town', 'village', 'kommun', 'stad']):
            category = 'cities'
        elif any(word in question_lower for word in ['mountain', 'peak', 'lake', 'forest', 'park', 'berg', 'sjö']):
            category = 'nature'
        elif any(word in question_lower for word in ['station', 'airport', 'hospital', 'school', 'universitet']):
            category = 'infrastructure'
        
        # Search the indexed data
        search_results = self.indexer.search(
            query=question,
            category=category,
            n_results=n_results
        )
        
        # Convert results to LangChain Documents
        retrieved_docs = []
        coordinates = {}
        
        for result in search_results:
            metadata = result['metadata'].copy()
            
            # Extract coordinates for potential use
            if 'epsg3006_x' in metadata and 'epsg3006_y' in metadata:
                if not coordinates:  # Use first result's coordinates as primary
                    coordinates = {
                        'x': metadata['epsg3006_x'],
                        'y': metadata['epsg3006_y']
                    }
            
            # Add score to metadata
            metadata['similarity_score'] = result['score']
            metadata['collection'] = result['collection']
            
            doc = Document(
                page_content=result['text'],
                metadata=metadata
            )
            retrieved_docs.append(doc)
        
        return {
            "context": retrieved_docs,
            "coordinates": coordinates if coordinates else state.get("coordinates")
        }
    
    def _generate(self, state: OSMRAGState) -> Dict[str, str]:
        """Generate answer using Gemini"""
        context_docs = state["context"]
        question = state["question"]
        
        # Prepare context text
        context_parts = []
        for doc in context_docs:
            metadata = doc.metadata
            content = doc.page_content
            
            # Add metadata information
            meta_info = []
            if 'name' in metadata:
                meta_info.append(f"Name: {metadata['name']}")
            if 'type' in metadata:
                meta_info.append(f"Type: {metadata['type']}")
            if 'population' in metadata:
                meta_info.append(f"Population: {metadata['population']:,}")
            if 'elevation' in metadata:
                meta_info.append(f"Elevation: {metadata['elevation']}m")
            if 'similarity_score' in metadata:
                meta_info.append(f"Relevance: {metadata['similarity_score']:.2f}")
            
            if meta_info:
                content = f"[{' | '.join(meta_info)}]\n{content}"
            
            context_parts.append(content)
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Generate response
        messages = self.prompt.invoke({
            "question": question, 
            "context": context_text
        })
        
        response = self.llm.invoke(messages)
        
        return {"answer": response.content}
    
    def query(self, 
              question: str, 
              location_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            location_filter: Optional filter for specific location types
            
        Returns:
            Dict containing answer, context, and metadata
        """
        initial_state = OSMRAGState(
            question=question,
            context=[],
            answer="",
            location_filter=location_filter,
            coordinates=None
        )
        
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "question": result["question"],
            "context_docs": result["context"],
            "coordinates": result.get("coordinates"),
            "num_sources": len(result["context"])
        }
    
    def chat(self):
        """Interactive chat interface"""
        print("🇸🇪 Swedish OpenStreetMap RAG Assistant")
        print("Ask me about Swedish places, cities, nature, and infrastructure!")
        print("Type 'quit' to exit, 'help' for guidance.\n")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if not question:
                    continue
                
                print("\n🔍 Searching Swedish OpenStreetMap data...")
                
                result = self.query(question)
                
                print(f"\n✅ **Answer** (based on {result['num_sources']} sources):")
                print(result["answer"])
                
                # Show coordinates if available
                if result.get("coordinates"):
                    coords = result["coordinates"]
                    print(f"\n📍 **Primary Location Coordinates (EPSG:3006):**")
                    print(f"   X (Easting): {coords['x']:,.0f}")
                    print(f"   Y (Northing): {coords['y']:,.0f}")
                
                # Show sources
                print(f"\n📚 **Sources used:**")
                for i, doc in enumerate(result["context_docs"][:3], 1):  # Show top 3
                    metadata = doc.metadata
                    name = metadata.get('name', 'Unknown')
                    location_type = metadata.get('type', 'Unknown type')
                    score = metadata.get('similarity_score', 0)
                    print(f"   {i}. {name} ({location_type}) - Relevance: {score:.2f}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with a different question.")
    
    def _show_help(self):
        """Show help information"""
        print("""
🆘 **Help - Swedish OpenStreetMap RAG Assistant**

**What you can ask:**
- "Where is Stockholm?" - Get information about Swedish cities
- "Tell me about Kebnekaise" - Learn about Swedish mountains and nature
- "What are the largest cities in Västergötland?" - Regional queries
- "Find hospitals in Uppsala" - Infrastructure and amenities
- "What's the elevation of Galdhøpiggen?" - Geographic details
- "Show me towns near coordinates X=674000 Y=6580000" - Coordinate-based search

**Coordinate System:**
- All coordinates are in EPSG:3006 (SWEREF99 TM) - Swedish national grid
- X = Easting (longitude-like), Y = Northing (latitude-like)

**Tips:**
- Ask in English or use Swedish place names
- Be specific about what you're looking for
- Questions about population, elevation, and location work well
- Try different phrasings if you don't get the expected results

**Commands:**
- 'quit' or 'exit' - Leave the chat
- 'help' - Show this help message
        """)


# Example usage and testing
def main():
    """Main function for testing the RAG system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Swedish OSM RAG Pipeline")
    parser.add_argument("--api-key", help="Google API key for Gemini")
    parser.add_argument("--db-path", default="./sweden_osm_rag", 
                        help="Path to ChromaDB database")
    parser.add_argument("--model", default="gemini-1.5-pro",
                        help="Gemini model to use")
    parser.add_argument("--chat", action="store_true",
                        help="Start interactive chat")
    parser.add_argument("--query", help="Single query to test")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        print("❌ ChromaDB database not found!")
        print(f"Please run OSM.py first to index OSM data at {args.db_path}")
        return
    
    try:
        # Initialize RAG system
        print("🚀 Initializing Swedish OSM RAG system...")
        rag = SwedishOSMRAG(
            db_path=args.db_path,
            google_api_key=args.api_key,
            model_name=args.model
        )
        print("✅ System ready!")
        
        if args.query:
            # Single query
            print(f"\n🔍 Query: {args.query}")
            result = rag.query(args.query)
            print(f"\n✅ Answer:\n{result['answer']}")
            
            if result.get("coordinates"):
                coords = result["coordinates"]
                print(f"\n📍 Coordinates (EPSG:3006): X={coords['x']:,.0f}, Y={coords['y']:,.0f}")
        
        elif args.chat:
            # Interactive chat
            rag.chat()
        
        else:
            # Demo queries
            demo_queries = [
                "Where is Stockholm and what is its population?",
                "Tell me about Kebnekaise mountain",
            ]
            
            print("\n🧪 Running demo queries...")
            for query in demo_queries:
                print(f"\n{'='*50}")
                print(f"Query: {query}")
                result = rag.query(query)
                print(f"Answer: {result['answer']}")
                print(f"Sources: {result['num_sources']}")
    
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        if "GOOGLE_API_KEY" in str(e):
            print("💡 Make sure to set your Google API key:")
            print("   export GOOGLE_API_KEY='your-api-key-here'")
            print("   or use --api-key argument")


if __name__ == "__main__":
    main()