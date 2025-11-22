import hashlib


def calculate_file_hash(content: bytes) -> str:
    """Calculates SHA-256 hash."""
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError("content must be bytes")
    return hashlib.sha256(bytes(content)).hexdigest()

