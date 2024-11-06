from dotenv import dotenv_values
from dspy.retrieve.pgvector_rm import PgVectorRM

from embedding_model import embedd

secret = dotenv_values('.secret')
db_url = secret['POSTGRES_URL']

def retriever_model(table_name: str) -> PgVectorRM:
    _retriever_model = PgVectorRM(
        db_url=db_url, 
        pg_table_name=table_name,
        k=1,
        embedding_func=embedd,
        embedding_field="embedding",
        fields=["text"],
        include_similarity=True
    )
    return _retriever_model