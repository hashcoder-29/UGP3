# db_utils.py
from sqlalchemy import create_engine
import os # Import the os library

# ## TODO: PASTE YOUR SUPABASE CONNECTION STRING HERE ##
# This local URL is now just a fallback for when you run the script on your laptop
DATABASE_URL_LOCAL = "postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxx.supabase.co:5432/postgres"

# Get the database URL from the GitHub Actions secret (environment variable)
# If it's not found, use the local URL as a backup.
DATABASE_URL = os.environ.get("DATABASE_URL", DATABASE_URL_LOCAL)

def get_db_engine():
    """Creates and returns a SQLAlchemy database engine."""
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return None