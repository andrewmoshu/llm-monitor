# Add a migration to add the context_window column if it doesn't exist
def add_context_window_column():
    from sqlalchemy import create_engine
    from sqlalchemy.sql import text
    
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE latency_records 
            ADD COLUMN IF NOT EXISTS context_window INTEGER
        """))
        conn.commit()

# Call this function when the application starts 