"""
Epstein Files RAG MCP Server
Main server implementation
"""

import os
import sys
import asyncio
from typing import Any, Sequence, List, Dict
from datetime import datetime

# Core dependencies
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, SearchRequest
import pandas as pd
import tiktoken

# MCP SDK
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.stdio import stdio_server

# Configuration
DATASET_NAME = "tensonaut/EPSTEIN_FILES_20K"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "epstein_files"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MAX_TOKENS_PER_RESULT = int(os.getenv("MAX_TOKENS_PER_RESULT", "150"))
MAX_QUERY_LIMIT = int(os.getenv("MAX_QUERY_LIMIT", "5"))
DEFAULT_QUERY_LIMIT = int(os.getenv("DEFAULT_QUERY_LIMIT", "3"))
BATCH_SIZE = 100


def log(message: str, level: str = "INFO"):
    """Simple logging to stderr with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", file=sys.stderr, flush=True)




def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Truncate text to a maximum number of tokens."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens) + "..."
    except Exception as e:
        log(f"Token truncation failed, using character fallback: {e}", "WARNING")
        # Fallback to character-based truncation (roughly 4 chars per token)
        return text[:max_tokens * 4] + "..."


class EpsteinRAGServer:
    """RAG server for Epstein Files dataset with Qdrant vector store."""

    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        self.collection_initialized = False
        log("EpsteinRAGServer instance created")

    def initialize(self):
        """Initialize embedding model and Qdrant client."""
        log("Starting initialization...")
        
        try:
            log(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            log("Embedding model loaded successfully")
        except Exception as e:
            log(f"Failed to load embedding model: {e}", "ERROR")
            raise

        try:
            log(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            log(f"Successfully connected to Qdrant! Found {len(collections.collections)} collections")
        except Exception as e:
            log(f"Failed to connect to Qdrant: {e}", "ERROR")
            log("Make sure Qdrant Docker container is running", "ERROR")
            raise

        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)

        if not collection_exists:
            log(f"Collection '{COLLECTION_NAME}' doesn't exist. Loading and indexing dataset...")
            self._load_and_index_dataset()
        else:
            log(f"Collection '{COLLECTION_NAME}' already exists. Ready to query.")
            self.collection_initialized = True
        
        log("Initialization complete!")

    def _load_and_index_dataset(self):
        """Load dataset from HuggingFace and index into Qdrant."""
        log(f"Loading dataset: {DATASET_NAME}")
        
        try:
            dataset = load_dataset(DATASET_NAME, split="train")
            df = pd.DataFrame(dataset)
            log(f"Dataset loaded successfully with {len(df)} rows")
        except Exception as e:
            log(f"Failed to load dataset: {e}", "ERROR")
            raise

        # Get embedding dimension
        try:
            sample_embedding = self.embedding_model.encode("test")
            embedding_dim = len(sample_embedding)
            log(f"Embedding dimension: {embedding_dim}")
        except Exception as e:
            log(f"Failed to get embedding dimension: {e}", "ERROR")
            raise

        # Create collection
        try:
            log(f"Creating Qdrant collection: {COLLECTION_NAME}")
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
            log("Collection created successfully")
        except Exception as e:
            log(f"Failed to create collection: {e}", "ERROR")
            raise

        # Prepare text for embedding
        log("Preparing documents for indexing...")
        documents = []
        for idx, row in df.iterrows():
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and isinstance(row[col], str):
                    text_parts.append(f"{col}: {row[col]}")
            doc_text = " | ".join(text_parts)
            documents.append({
                'id': idx,
                'text': doc_text,
                'metadata': row.to_dict()
            })
        
        log(f"Prepared {len(documents)} documents")

        # Index in batches
        log(f"Indexing {len(documents)} documents in batches of {BATCH_SIZE}")
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_num, i in enumerate(range(0, len(documents), BATCH_SIZE), 1):
            try:
                batch = documents[i:i + BATCH_SIZE]
                texts = [doc['text'] for doc in batch]
                
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

                points = [
                    PointStruct(
                        id=doc['id'],
                        vector=embeddings[j].tolist(),
                        payload={
                            'text': doc['text'],
                            **doc['metadata']
                        }
                    )
                    for j, doc in enumerate(batch)
                ]

                self.qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )

                if batch_num % 10 == 0:
                    log(f"Indexed {i + BATCH_SIZE}/{len(documents)} documents ({batch_num}/{total_batches} batches)")
            
            except Exception as e:
                log(f"Failed to index batch {batch_num}: {e}", "ERROR")
                raise

        log("Indexing complete!")
        self.collection_initialized = True

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the vector store for relevant documents."""
        log(f"Search - query: '{query[:50]}...', limit: {limit}")
        
        if not self.collection_initialized:
            log("Collection not initialized", "ERROR")
            raise RuntimeError("Collection not initialized")

        try:
            # Encode query
            query_vector = self.embedding_model.encode(query).tolist()
            
            # Use query_points method (the correct method name in newer qdrant-client)
            try:
                search_result = self.qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=limit
                ).points
            except AttributeError:
                # Fallback for older versions that use search()
                log("query_points not available, trying search()", "WARNING")
                search_result = self.qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=limit
                )
            
            log(f"Qdrant returned {len(search_result)} results")

            # Format results
            results = []
            for hit in search_result:
                results.append({
                    'score': hit.score,
                    'text': hit.payload.get('text', ''),
                    'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
                })

            log(f"Search successful, returning {len(results)} results")
            return results
        
        except Exception as e:
            log(f"Search failed: {e}", "ERROR")
            raise


# Initialize the RAG server
rag_server = EpsteinRAGServer()

# Create MCP server
mcp_server = Server("epstein-rag-server")
log("MCP server created")


@mcp_server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    log("list_tools() called")
    return [
        Tool(
            name="search_epstein_files",
            description="Search through the Epstein Files dataset using semantic search. Returns the most relevant documents based on the query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents"
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Maximum number of results to return (default: {DEFAULT_QUERY_LIMIT}, max: {MAX_QUERY_LIMIT})",
                        "default": DEFAULT_QUERY_LIMIT,
                        "minimum": 1,
                        "maximum": MAX_QUERY_LIMIT
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": f"Maximum tokens per result excerpt (default: {MAX_TOKENS_PER_RESULT})",
                        "default": MAX_TOKENS_PER_RESULT,
                        "minimum": 50,
                        "maximum": 1000
                    }
                },
                "required": ["query"]
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    log("=" * 80)
    log(f"TOOL CALL: {name}")
    log(f"Arguments: {arguments}")
    
    if name != "search_epstein_files":
        log(f"Unknown tool: {name}", "ERROR")
        raise ValueError(f"Unknown tool: {name}")

    try:
        # Extract arguments
        query = arguments.get("query") if isinstance(arguments, dict) else None
        limit = arguments.get("limit", DEFAULT_QUERY_LIMIT) if isinstance(arguments, dict) else DEFAULT_QUERY_LIMIT
        max_tokens_per_result = arguments.get("max_tokens", MAX_TOKENS_PER_RESULT) if isinstance(arguments, dict) else MAX_TOKENS_PER_RESULT
        
        # Cap limit to configured maximum
        original_limit = limit
        limit = min(limit, MAX_QUERY_LIMIT)
        if original_limit != limit:
            log(f"Limit capped from {original_limit} to {limit} (max: {MAX_QUERY_LIMIT})", "WARNING")
        
        log(f"Query: '{query}', limit: {limit}, max_tokens: {max_tokens_per_result}")

        if not query:
            log("Query parameter missing", "ERROR")
            raise ValueError("Query parameter is required")
        
        # Perform search
        start_time = datetime.now()
        results = await asyncio.to_thread(rag_server.search, query, limit)
        elapsed = (datetime.now() - start_time).total_seconds()
        log(f"Search completed in {elapsed:.2f}s")

        # Format response
        response_text = f"Found {len(results)} relevant documents:\n\n"
        
        for i, result in enumerate(results, 1):
            text_excerpt = truncate_to_tokens(result['text'], max_tokens_per_result)
            response_text += f"{i}. [Score: {result['score']:.3f}]\n{text_excerpt}\n\n"

        log(f"TOOL CALL SUCCESS - returned {len(results)} results")
        log("=" * 80)
        
        return [TextContent(type="text", text=response_text)]
    
    except Exception as e:
        log(f"TOOL CALL FAILED: {type(e).__name__}: {str(e)}", "ERROR")
        log("=" * 80)
        
        # Print full traceback
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        return [TextContent(type="text", text=f"Search failed: {str(e)}")]


async def main():
    """Main entry point for the MCP server."""
    try:
        log("Starting RAG server initialization...")
        rag_server.initialize()
        log("RAG server ready!")
    except Exception as e:
        log(f"FATAL: Initialization failed: {e}", "ERROR")
        raise

    log("Starting MCP server on stdio...")
    async with stdio_server() as (read_stream, write_stream):
        log("MCP server running - ready for requests")
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


def main_sync():
    """Synchronous entry point for console script."""
    import argparse


    parser = argparse.ArgumentParser(description="Epstein Files RAG MCP Server")
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"),
                       help="Qdrant host (default: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")),
                       help="Qdrant port (default: 6333)")
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS_PER_RESULT", "150")),
                       help="Maximum tokens per result excerpt (default: 150)")
    parser.add_argument("--default-limit", type=int, default=int(os.getenv("DEFAULT_QUERY_LIMIT", "3")),
                       help="Default number of search results (default: 3)")
    parser.add_argument("--max-limit", type=int, default=int(os.getenv("MAX_QUERY_LIMIT", "5")),
                       help="Maximum number of search results allowed (default: 5)")
    args = parser.parse_args()

    # Override global config with CLI args
    global QDRANT_HOST, QDRANT_PORT, MAX_TOKENS_PER_RESULT, DEFAULT_QUERY_LIMIT, MAX_QUERY_LIMIT
    QDRANT_HOST = args.qdrant_host
    QDRANT_PORT = args.qdrant_port
    MAX_TOKENS_PER_RESULT = args.max_tokens
    DEFAULT_QUERY_LIMIT = args.default_limit
    MAX_QUERY_LIMIT = args.max_limit

    log(f"Config: QDRANT_HOST={QDRANT_HOST}, QDRANT_PORT={QDRANT_PORT}, MAX_TOKENS={MAX_TOKENS_PER_RESULT}, DEFAULT_LIMIT={DEFAULT_QUERY_LIMIT}, MAX_LIMIT={MAX_QUERY_LIMIT}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Shutdown by user")
    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
