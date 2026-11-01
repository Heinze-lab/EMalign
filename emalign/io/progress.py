import datetime
from pymongo import MongoClient
import json

def get_mongo_client(config_file=None):
    '''
    Returns a MongoClient object.
    If config_file is provided, it will be used to configure the client.
    Otherwise, it will use default parameters.
    '''
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return MongoClient(**config)
    else:
        return MongoClient(None)

def get_mongo_db(client, project_name):
    '''
    Returns a database object.
    '''
    db_name = f'alignment_{project_name}'
    return client[db_name]

def log_progress(db, collection_name, step_name, global_slice_index, local_slice_index, metadata):
    '''
    Logs progress to the database.
    '''
    collection = db[collection_name]
    doc = {
        'step_name': step_name,
        'global_slice': global_slice_index,
        'local_slice': local_slice_index,
        'timestamp': datetime.datetime.now(datetime.UTC),
        **metadata
    }
    collection.insert_one(doc)

def check_progress(db, collection_name, step_name, local_slice_index, doc_filter={}, return_timestamp=False):
    '''
    Checks if a slice has already been processed.
    If return_timestamp, also returns the time it was last processed (None if it was not).
    '''
    collection = db[collection_name]
    doc_filter = doc_filter | {'step_name': step_name, 'local_slice': local_slice_index}
    if not return_timestamp:
        return collection.count_documents(doc_filter) > 0
    doc = collection.find_one(doc_filter, sort=[('timestamp', -1)])
    return doc is not None, (doc.get('timestamp') if doc is not None else None)

def format_timestamp(timestamp):
    '''
    Formats a timestamp from the database in local time, e.g. 26.10.17 - 14:05.
    '''
    if timestamp is None:
        return 'unknown date'
    # Timestamps are stored in UTC, pymongo returns them without timezone
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=datetime.UTC)
    return timestamp.astimezone().strftime('%y.%m.%d - %H:%M')

def wipe_progress(db, collection_name, step_name=None):
    '''
    Wipes the progress for a given stack.
    '''
    if step_name is None:
        db.drop_collection(collection_name)
    else:
        db[collection_name].delete_many({'step_name': step_name})