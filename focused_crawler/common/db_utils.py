import os
import sqlite3
from contextlib import closing
from typing import List, Tuple

import numpy as np

# Every site of a run is crawled in its own process, and the processes share these
# SQLite files. Each call opens and closes its own connection, so no connection is
# shared between threads or processes. SQLite's file locks serialise the writes of
# concurrent processes; they are reliable on a local disk, not on a network file system.

# Seconds a connection waits for a lock held by another process before it fails with
# "database is locked" (sqlite3's default is 5).
SQLITE_TIMEOUT = 60


def get_label(url: str, db_dir: str) -> bool:
    db_connection = sqlite3.connect(db_dir, timeout=SQLITE_TIMEOUT)
    cursor = db_connection.cursor()

    rows = list(
        cursor.execute(
            f"SELECT * FROM labels WHERE url = ?",
            (url,),
        ),
    )

    cursor.close()
    db_connection.commit()
    db_connection.close()

    try:
        label = rows[0][1] == "True"
    except IndexError:
        label = False

    return label


def get_all_positive_labels_for_domain(domain: str, db_dir: str) -> List[bool]:
    db_connection = sqlite3.connect(db_dir, timeout=SQLITE_TIMEOUT)
    cursor = db_connection.cursor()

    rows = list(
        cursor.execute(f"SELECT * FROM labels WHERE url like ?", (f"%{domain}%",))
    )

    cursor.close()
    db_connection.commit()
    db_connection.close()

    rows = [bool(r[1]) for r in rows if r[1] == "True"]
    return rows


def get_embedding_from_cache(anchor_text: str, cache_dir: str) -> List[float]:
    with closing(sqlite3.connect(cache_dir, timeout=SQLITE_TIMEOUT)) as db_connection:
        cursor = db_connection.cursor()

        try:
            # 1. Execute the query
            cursor.execute(
                "SELECT embedding FROM embeddings WHERE anchor_text = ?",
                (anchor_text,),
            )
            row = cursor.fetchone()

            # 2. Handle cache miss instantly and cleanly
            if not row:
                return []

            # 3. Extract the binary blob
            blob = row[0]
            if blob:
                embedding_array = np.frombuffer(blob, dtype=np.float32)
                return embedding_array.tolist()

            return []

        finally:
            cursor.close()


def save_embedding_into_cache(
    anchor_text: str,
    embedding: List[float],
    cache_dir: str,
):

    embedding_array = np.array(
        embedding,
        dtype=np.float32,
    )
    embedding_blob = embedding_array.tobytes()

    # Two processes can miss the same anchor text and both insert it; the UNIQUE
    # constraint rejects the second insert.
    with closing(sqlite3.connect(cache_dir, timeout=SQLITE_TIMEOUT)) as db_connection:
        cursor = db_connection.cursor()

        try:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (anchor_text TEXT UNIQUE NOT NULL, embedding BLOB NOT NULL)"
            )

            cursor.execute(
                "INSERT INTO embeddings(anchor_text, embedding) VALUES (?, ?);",
                (
                    anchor_text,
                    embedding_blob,
                ),
            )

            db_connection.commit()

        except sqlite3.IntegrityError:
            print(f"--> Anchor text {anchor_text} was already in DB!")

        finally:
            cursor.close()


def save_judgment(
    link: str,
    anchor_text:str,
    score: float,
    label: bool,
    cache_dir: str,
    cache_file: str,
):

    # Create directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # All the sites of a run log to the same file. sqlite3.connect creates it without
    # emptying a file that another process has just created.
    file_path = os.path.join(cache_dir, cache_file)
    is_new_file = not os.path.exists(file_path)

    db_connection = sqlite3.connect(file_path, timeout=SQLITE_TIMEOUT)
    if is_new_file:
        os.chmod(file_path, 0o777)
    cursor = db_connection.cursor()

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS judgments (link text UNIQUE NOT NULL, anchor_text text NOT NULL, score real NOT NULL, label integer NOT NULL)"
    )

    try:
        cursor.execute(
            "INSERT INTO judgments(link, anchor_text, score, label) VALUES (?, ?, ?, ?);",
            (link, anchor_text, score, int(label)),
        )

    except sqlite3.IntegrityError:
        print(f"--> Link {link} was already in DB!")

    cursor.close()
    db_connection.commit()
    db_connection.close()


def get_judgment(url: str, path: str) -> Tuple[float, int]:
    db_connection = sqlite3.connect(path, timeout=SQLITE_TIMEOUT)
    cursor = db_connection.cursor()

    rows = list(
        cursor.execute(
            f"SELECT score, label FROM judgments WHERE link = ?",
            (url,),
        ),
    )

    cursor.close()
    db_connection.commit()
    db_connection.close()

    return rows[0]


def get_all_judgments(path: str) -> List[Tuple[float, int]]:
    db_connection = sqlite3.connect(path, timeout=SQLITE_TIMEOUT)
    cursor = db_connection.cursor()

    rows = list(
        cursor.execute(f"SELECT score, label FROM judgments"),
    )

    cursor.close()
    db_connection.commit()
    db_connection.close()

    return rows


def bulk_judgments(
    judgments: List[Tuple],
    cache_dir: str,
    cache_file: str,
):

    # Create directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # All the sites of a run log to the same file. sqlite3.connect creates it without
    # emptying a file that another process has just created.
    file_path = os.path.join(cache_dir, cache_file)
    is_new_file = not os.path.exists(file_path)

    db_connection = sqlite3.connect(file_path, timeout=SQLITE_TIMEOUT)
    if is_new_file:
        os.chmod(file_path, 0o777)
    cursor = db_connection.cursor()

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS judgments (link text UNIQUE NOT NULL, anchor_text text NOT NULL, score real NOT NULL, label integer NOT NULL)"
    )

    try:
        cursor.executemany(
            "INSERT INTO judgments(link, anchor_text, score, label) VALUES (?, ?, ?, ?);",
            judgments,
        )

    except sqlite3.IntegrityError:
        print(f"--> Link {judgments} was already in DB!")

    cursor.close()
    db_connection.commit()
    db_connection.close()
