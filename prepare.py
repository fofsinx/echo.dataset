import os
import uuid
import csv
import concurrent.futures
from faster_whisper import WhisperModel
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Whisper model with GPU
model = WhisperModel("large-v2", device="cuda", compute_type="float16")

def transcribe_audio(row):
    """Transcribe a single audio file using Whisper model"""
    try:
        logger.info(f"🎯 File: prepare.py, Line: 16, Function: transcribe_audio, Processing file: {row['path']}")
        segments, _ = model.transcribe(row['path'])
        transcription = " ".join([segment.text for segment in segments])
        return {
            'id': row['id'],
            'path': row['path'],
            'transcription': transcription
        }
    except Exception as e:
        logger.error(f"❌ Error processing {row['path']}: {str(e)}")
        return {
            'id': row['id'], 
            'path': row['path'],
            'transcription': ''
        }

def process_csv_file(csv_path):
    """Process a CSV file and add transcriptions"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"📊 File: prepare.py, Line: 35, Function: process_csv_file, Processing CSV: {csv_path}")

        # Create ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Process each row in parallel
            results = list(executor.map(transcribe_audio, df.to_dict('records')))

        # Create new DataFrame with results
        new_df = pd.DataFrame(results)
        
        # Save back to CSV
        new_df.to_csv(csv_path, index=False)
        logger.info(f"✅ Successfully processed {csv_path}")

    except Exception as e:
        logger.error(f"❌ Error processing CSV {csv_path}: {str(e)}")

def get_folders(parent_path):
    return [os.path.join(parent_path, f) for f in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, f))]

def create_csv_for_folder(folder_path):
    folder_name = os.path.basename(folder_path.rstrip("/"))
    csv_filename = f"{folder_name}_dataset.csv"
    
    data = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            data.append({"id": str(uuid.uuid4()), "path": file_path})
    
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "path"])
        writer.writeheader()
        writer.writerows(data)
    
    logger.info(f"📝 Created CSV for folder '{folder_name}': {csv_filename}")
    
    # Process the newly created CSV to add transcriptions
    process_csv_file(csv_filename)

# Main execution
if __name__ == "__main__":
    base_path = "/Users/harshvardhangoswami/Projects/airley.ai/tts/data_preparation/audios"
    for folder in get_folders(base_path):
        create_csv_for_folder(folder)
