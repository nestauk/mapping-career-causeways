import urllib.request
import os
import zipfile
import sys

"""
Usage
-----

Option 1, for downloading all data files, to replicate all analyses:
$ python download_data.py

Option 2, for downloading only data required for generating transitions:
$ python download_data.py lite
"""

# Choose which archive to download
if (len(sys.argv) > 1) & (sys.argv[1]=='lite'):
    # Lite archive, only with required inputs for generating and analysing transitions
    archive_file = 'data_lite.zip'
    file_size_mb = 300
else:
    # Full archive with all raw, interim and processed data
    archive_file = 'data.zip'
    file_size_mb = 600

fpath = ''
path_to_zip_file = fpath + archive_file
url_to_zip_file = f'https://ojd-mapping-career-causeways.s3.eu-west-2.amazonaws.com/{archive_file}'

directory_to_extract_to = ''

# Check if any of the folders already exist
if (os.path.exists('raw/') or \
    os.path.exists('interim/') or \
    os.path.exists('processed/')):

    print(f"Some data apears to already exist! To replace the data folders, please first delete the 'raw', 'interim' and 'processed' folders")

else:
    # If archive file doesn't exist, download from S3
    if os.path.exists(path_to_zip_file)==False:
        print(f'Downloading {path_to_zip_file} (approx. {file_size_mb} MB)...', end=' ')
        urllib.request.urlretrieve(url_to_zip_file, path_to_zip_file)
        print('Done!')

    # Extract the archive
    print(f'Extracting the archive in {directory_to_extract_to}...', end=' ')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print('Done!')
