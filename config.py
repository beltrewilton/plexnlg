from dotenv import dotenv_values
from dspy.retrieve.pgvector_rm import PgVectorRM

from embedding_model import embedd

secret = dotenv_values('.secret')


def retriever_model(node: str, table_name: str) -> PgVectorRM:
    print("secret[node]")
    print(table_name, secret[node])
    
    _retriever_model = PgVectorRM(
        db_url=secret[node], 
        pg_table_name=table_name,
        k=1,
        embedding_func=embedd,
        embedding_field="embedding",
        fields=["text"],
        include_similarity=True
    )
    return _retriever_model