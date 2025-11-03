# options_archiver.py (Corrected)
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from db_utils import get_db_engine

def archive_and_cleanup_to_db():
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    today_str = today.strftime('%Y-%m-%d')
    # ## --- FIX --- ##
    # Define the 'yesterday_str' variable that was missing.
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    aggregated_file_today = f"nifty_options_aggregated_{today_str}.csv"
    
    # --- Archiving Step ---
    try:
        logging.info(f"Reading daily aggregated data from {aggregated_file_today}...")
        daily_df = pd.read_csv(aggregated_file_today)
        daily_df['date'] = today_str
        
        engine = get_db_engine()
        if engine is not None:
            master_table_name = "historical_options"
            logging.info(f"Appending data to database table: {master_table_name}")
            
            daily_df.to_sql(master_table_name, engine, if_exists='append', index=False)
            logging.info("✅ Archiving to database successful.")
        else:
            raise Exception("Could not connect to the database.")

    except FileNotFoundError:
        logging.error(f"❌ Archiving failed: Input file not found: {aggregated_file_today}")
        return
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during archiving: {e}")
        return

    # --- Cleanup Step ---
    # logging.info("Starting cleanup of daily files...")
    # files_to_delete = [
    #     f"nifty_options_{today_str}.csv",
    #     f"nifty_options_{yesterday_str}.csv",
    #     aggregated_file_today
    # ]
    
    # for file_path in files_to_delete:
    #     try:
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #             logging.info(f"  - Deleted: {file_path}")
    #     except OSError as e:
    #         logging.error(f"  - Error deleting file {file_path}: {e}")
            
    # logging.info("✅ Cleanup successful.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("--- Starting DB Archiver and Cleanup Script ---")
    archive_and_cleanup_to_db()
    print("--- DB Archiver and Cleanup Finished ---")