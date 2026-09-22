import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    brand TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'placed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS semantic_memories (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    content TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS episodic_memories (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_id INTEGER REFERENCES orders(id),
    event TEXT NOT NULL,
    outcome TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS procedures (
    name TEXT PRIMARY KEY,
    steps JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

SEED_CUSTOMERS = ["Alice", "Bob"]

SEED_PRODUCTS = [
    ("Nike Air Max", "Nike", "lifestyle", 129.99,
     "Cushioned everyday sneaker with visible Air unit, moderate weight."),
    ("Nike Pegasus", "Nike", "running", 134.99,
     "Versatile daily running shoe, responsive cushioning, standard width fit."),
    ("Adidas Ultraboost", "Adidas", "running", 189.99,
     "Energy-returning Boost cushioning, snug sock-like knit upper, great for long runs."),
    ("New Balance 530", "New Balance", "lifestyle", 99.99,
     "Retro-styled lightweight sneaker, casual everyday wear."),
    ("Brooks Ghost", "Brooks", "running", 139.99,
     "Smooth, balanced ride, wide toe box, popular for high weekly mileage."),
    ("Hoka Clifton", "Hoka", "running", 144.99,
     "Ultra-lightweight max-cushion running shoe, ideal for long distances."),
]

SEED_PROCEDURES = {
    "recommend_running_shoes": [
        "Ask about the customer's running distance, if not already known.",
        "Ask whether they prefer lightweight or cushioned shoes, if not already known.",
        "Check the customer's previous purchases.",
        "Check the customer's previous returns.",
        "Recommend 2-3 products.",
    ],
}


def get_conn() -> psycopg.Connection:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")
    conn = psycopg.connect(database_url, row_factory=dict_row)
    register_vector(conn)
    return conn


def init_db() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")

    # register_vector() needs the "vector" type to already exist, so create the
    # extension on a plain connection before any get_conn() call registers it.
    with psycopg.connect(database_url) as bootstrap_conn:
        bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        bootstrap_conn.commit()

    with get_conn() as conn:
        conn.execute(SCHEMA)
        for name in SEED_CUSTOMERS:
            conn.execute(
                "INSERT INTO customers (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                (name,),
            )
        for name, brand, category, price, description in SEED_PRODUCTS:
            conn.execute(
                """
                INSERT INTO products (name, brand, category, price, description)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING
                """,
                (name, brand, category, price, description),
            )
        for name, steps in SEED_PROCEDURES.items():
            conn.execute(
                "INSERT INTO procedures (name, steps) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                (name, Jsonb(steps)),
            )
        conn.commit()


def get_or_create_customer(name: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, name FROM customers WHERE name ILIKE %s", (name,)
        ).fetchone()
        if row:
            return row
        row = conn.execute(
            "INSERT INTO customers (name) VALUES (%s) RETURNING id, name", (name,)
        ).fetchone()
        conn.commit()
        return row


def search_products(query: str) -> list[dict]:
    words = query.split() or [query]
    conditions = []
    params: dict[str, str] = {}
    for i, word in enumerate(words):
        key = f"w{i}"
        conditions.append(
            f"(name ILIKE %({key})s OR brand ILIKE %({key})s "
            f"OR category ILIKE %({key})s OR description ILIKE %({key})s)"
        )
        params[key] = f"%{word}%"

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT id, name, brand, category, price, description
            FROM products
            WHERE {" OR ".join(conditions)}
            ORDER BY name
            """,
            params,
        ).fetchall()
        return rows


def get_product(product_id: int) -> dict | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, name, brand, category, price, description FROM products WHERE id = %s",
            (product_id,),
        ).fetchone()


def create_order(customer_id: int, product_id: int, quantity: int = 1) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO orders (customer_id, product_id, quantity)
            VALUES (%s, %s, %s)
            RETURNING id, customer_id, product_id, quantity, status, created_at
            """,
            (customer_id, product_id, quantity),
        ).fetchone()
        conn.commit()
        return row


def add_semantic_memory(customer_id: int, content: str, embedding: list[float]) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO semantic_memories (customer_id, content, embedding)
            VALUES (%s, %s, %s)
            RETURNING id, customer_id, content, created_at
            """,
            (customer_id, content, embedding),
        ).fetchone()
        conn.commit()
        return row


def search_semantic_memories(customer_id: int, query_embedding: list[float], k: int = 3) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT content, embedding <=> %(q)s::vector AS distance
            FROM semantic_memories
            WHERE customer_id = %(customer_id)s
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
            """,
            {"q": query_embedding, "customer_id": customer_id, "k": k},
        ).fetchall()


def get_order(order_id: int) -> dict | None:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT o.id, o.customer_id, o.product_id, o.quantity, o.status, o.created_at,
                   c.name AS customer_name,
                   p.name AS product_name, p.brand, p.price
            FROM orders o
            JOIN customers c ON c.id = o.customer_id
            JOIN products p ON p.id = o.product_id
            WHERE o.id = %s
            """,
            (order_id,),
        ).fetchone()


def update_order_status(order_id: int, status: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            UPDATE orders SET status = %s WHERE id = %s
            RETURNING id, customer_id, product_id, quantity, status, created_at
            """,
            (status, order_id),
        ).fetchone()
        conn.commit()
        return row


def add_episodic_memory(
    customer_id: int, order_id: int | None, event: str, outcome: str, embedding: list[float]
) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO episodic_memories (customer_id, order_id, event, outcome, embedding)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, customer_id, order_id, event, outcome, created_at
            """,
            (customer_id, order_id, event, outcome, embedding),
        ).fetchone()
        conn.commit()
        return row


def search_episodic_memories(customer_id: int, query_embedding: list[float], k: int = 3) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT event, outcome, embedding <=> %(q)s::vector AS distance
            FROM episodic_memories
            WHERE customer_id = %(customer_id)s
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
            """,
            {"q": query_embedding, "customer_id": customer_id, "k": k},
        ).fetchall()


def get_procedure(name: str) -> dict | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT name, steps, updated_at FROM procedures WHERE name = %s",
            (name,),
        ).fetchone()


def update_procedure(name: str, steps: list[str]) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO procedures (name, steps, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (name) DO UPDATE SET steps = EXCLUDED.steps, updated_at = now()
            RETURNING name, steps, updated_at
            """,
            (name, Jsonb(steps)),
        ).fetchone()
        conn.commit()
        return row
