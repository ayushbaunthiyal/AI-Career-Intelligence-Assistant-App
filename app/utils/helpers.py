"""Utility helper functions."""

import hashlib
import uuid
from pathlib import Path


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1.5 MB").
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension.

    Args:
        filename: Name of the file.

    Returns:
        Lowercase extension including the dot (e.g., ".pdf").
    """
    return Path(filename).suffix.lower()


def generate_doc_id(content: str, doc_type: str) -> str:
    """Generate a unique document ID based on content hash.

    Args:
        content: Document text content.
        doc_type: Type of document ("resume" or "job_posting").

    Returns:
        Unique document ID string.
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{doc_type}_{content_hash}"


def generate_chunk_id() -> str:
    """Generate a unique chunk ID.

    Returns:
        UUID string for chunk identification.
    """
    return str(uuid.uuid4())


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename safe for filesystem.
    """
    # Remove path separators and null bytes
    sanitized = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")
    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        sanitized = name[:255 - len(ext)] + ext
    return sanitized
