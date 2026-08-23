import re

from utils.hashing import generate_analysis_id, md5_bytes, sha256_bytes


def test_sha256_matches_known_vector():
    # sha256("") is a fixed, well-known value — a good sanity check that
    # we're hashing raw bytes correctly and not, say, a str() of the bytes.
    assert sha256_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_is_deterministic_and_sensitive_to_content():
    a = sha256_bytes(b"hello world")
    b = sha256_bytes(b"hello world")
    c = sha256_bytes(b"hello world!")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_md5_is_deterministic_and_correct_length():
    a = md5_bytes(b"hello world")
    b = md5_bytes(b"hello world")
    assert a == b
    assert len(a) == 32


def test_analysis_id_format():
    aid = generate_analysis_id()
    assert re.match(r"^FD-\d{8}-[0-9A-F]{8}$", aid), aid


def test_analysis_id_is_unique_across_calls():
    ids = {generate_analysis_id() for _ in range(50)}
    assert len(ids) == 50
