import sqlite3
import pandas as pd
from pathlib import Path
import argparse

def view_database(market='UK'):
    """View the contents of the SQLite database"""
    # Get database path
    market_dir = Path(__file__).parent / 'data' / market.lower()
    db_path = market_dir / f'{market.lower()}_market.db'
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    
    # Get list of tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\nDatabase: {db_path}")
    print(f"Tables found: {[table[0] for table in tables]}\n")
    
    # View each table
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        print("-" * 50)
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print("\nColumns:")
        for col in columns:
            print(f"- {col[1]} ({col[2]})")
        
        # Get sample data
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print("\nSample data:")
        print(df)
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View database contents')
    parser.add_argument('--market', type=str, default='UK', help='Market to view (UK or US)')
    args = parser.parse_args()
    
    view_database(args.market) 