import argparse
import asyncio
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any

from chonkie import CodeChunker
from qdrant_client import AsyncQdrantClient, models

from dotenv import load_dotenv

load_dotenv()

from . import services


LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".cs": "c_sharp", ".rb": "ruby", ".php": "php", ".html": "html",
    ".css": "css", ".json": "json", ".md": "markdown", ".sh": "bash",
    ".yaml": "yaml", ".yml": "yaml",
}

# Files/directories to ignore
IGNORE_PATTERNS = [
    ".git", "__pycache__", "node_modules", "target", "build", "dist",
    ".venv", ".env", "venv", "migrations", "alembic",
]

# File extensions to ignore
IGNORE_EXTENSIONS = [
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
    ".mp4", ".mov", ".avi", ".zip", ".tar", ".gz", ".pdf",
    ".doc", ".docx", ".xls", ".xlsx", ".lock", ".log",
    ".sum", ".mod", ".work", ".idea", ".vscode", ".DS_Store"
]

def is_ignored(path: Path) -> bool:
    """Check if a file or directory should be ignored."""
    if path.name.startswith('.') or path.name.startswith('_'):
        return True
    if path.suffix.lower() in IGNORE_EXTENSIONS:
        return True
    return any(part in IGNORE_PATTERNS for part in path.parts)

async def get_file_chunks(file_path: Path, language: str) -> List[Dict[str, Any]]:
    """Chunks a single file and returns a list of chunk data."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except (IOError, UnicodeDecodeError) as e:
        print(f"Could not read file {file_path}: {e}")
        return []

    if not code.strip():
        return []

    chunker = CodeChunker(language=language)
    chunks = []
    try:
        code_chunks = chunker.chunk(code)
        for chunk in code_chunks:
            chunks.append({
                "text": chunk.text,
                "language": language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            })
    except Exception as e:
        print(f"Error chunking file {file_path} with language {language}: {e}")
    
    return chunks


async def process_repository(repo_path: Path, qdrant_client: AsyncQdrantClient, collection_name: str):
    """Walks a repository, chunks files, and upserts them to Qdrant."""
    tasks = []
    for root, _, files in os.walk(repo_path):
        root_path = Path(root)
        if is_ignored(root_path):
            continue
            
        for filename in files:
            file_path = root_path / filename
            if is_ignored(file_path):
                continue

            language = LANGUAGE_MAP.get(file_path.suffix.lower())
            if not language:
                continue

            tasks.append(process_file(file_path, repo_path, language, qdrant_client, collection_name))

    await asyncio.gather(*tasks)

async def process_file(file_path: Path, repo_path: Path, language: str, qdrant_client: AsyncQdrantClient, collection_name: str):
    """Process a single file: chunk, summarize, embed, and upsert."""
    relative_path = file_path.relative_to(repo_path)
    print(f"Processing: {relative_path}")
    
    chunks = await get_file_chunks(file_path, language)
    if not chunks:
        return

    points_to_upsert = []
    for chunk in chunks:
        chunk_text = chunk["text"]
        
        summary_task = services.summarize_code_chunk(chunk_text, str(relative_path), chunk["language"])
        embedding_task = services.get_embedding(chunk_text)

        summary, embedding = await asyncio.gather(summary_task, embedding_task)
        
        if not embedding:
            continue

        payload = {
            "text": chunk_text,
            "language": chunk["language"],
            "file_path": str(relative_path),
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "summary": summary,
        }

        points_to_upsert.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
        )

    if points_to_upsert:
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=True,
        )
        print(f"Upserted {len(points_to_upsert)} points from {relative_path}")

async def main():
    parser = argparse.ArgumentParser(description="Chunk a GitHub repository and store it in Qdrant.")
    parser.add_argument("repo_url", help="The URL of the GitHub repository to process.")
    parser.add_argument("--qdrant-url", help="URL for Qdrant instance. Overrides QDRANT_URL from .env")
    parser.add_argument("--collection-name", help="Name of the Qdrant collection. Defaults to repo name.", default="chunkrant")
    parser.add_argument("--embedding-dim", type=int, default=768, help="Dimension of the embeddings (e.g., nomic-embed-text is 768).")
    
    args = parser.parse_args()
    
    repo_url = args.repo_url
    embedding_dim = args.embedding_dim

    # Get Qdrant connection details from args or environment variables
    # This prioritizes the command-line argument, then .env file, then a local default.
    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL") or "http://localhost:6333"
    api_key = os.getenv("QDRANT_API_KEY")  # API key should only come from env


    collection_name = args.collection_name or repo_url.split("/")[-1].replace(".git", "")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Cloning {repo_url} into {temp_path}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, "."],
                cwd=temp_path, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr}")
            return
        
        print("Clone successful.")

        qdrant_client = None
        try:
            print(f"Connecting to Qdrant at {qdrant_url}...")
            qdrant_client = AsyncQdrantClient(url=qdrant_url, api_key=api_key, timeout=30)

            
            # Check if collection exists
            collection_info = await qdrant_client.get_collection(collection_name=collection_name)
            existing_dim = collection_info.vectors_config.params.size
            if existing_dim != embedding_dim:
                print(f"Error: Collection '{collection_name}' exists with dimension {existing_dim}, but requested dimension is {embedding_dim}.")
                print("Please delete the collection or use the correct embedding dimension.")
                return
            print(f"Using existing collection '{collection_name}'.")

        except Exception as e:
            # Handle case where collection doesn't exist
            if "not found" in str(e).lower() or ("status_code=404" in str(e)):
                 print(f"Collection '{collection_name}' not found. Creating...")
                 try:
                    await qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
                    )
                    print("Collection created.")
                 except Exception as create_exc:
                    print(f"Failed to create collection: {create_exc}")
                    return
            else:
                print(f"Error connecting to or setting up Qdrant: {e}")
                print("\nPlease check your QDRANT_URL and QDRANT_API_KEY in your .env file or command line arguments.")
                print("For cloud instances, the URL should not contain a port (e.g., https://<...>.cloud.qdrant.io).")
                return

        try:
            await process_repository(temp_path, qdrant_client, collection_name)
            print("\nFinished processing repository.")
        except Exception as e:
            print(f"An error occurred during repository processing: {e}")
        finally:
            if qdrant_client:
                await qdrant_client.close()

if __name__ == "__main__":
    asyncio.run(main()) 