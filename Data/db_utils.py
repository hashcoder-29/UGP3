# db_utils.py
from sqlalchemy import create_engine

# ## TODO: PASTE YOUR SUPABASE CONNECTION STRING HERE ##
DATABASE_URL = "postgresql://postgres:NwTZ7H2#qTQLv2.@db.pcmrcihxokxyrzodvdxz.supabase.co:5432/postgres"

def get_db_engine():
    """Creates and returns a SQLAlchemy database engine."""
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return None