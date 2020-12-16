import urllib.request
import os
import zipfile

fpath = ''
path_to_zip_file = fpath + 'data.zip'
directory_to_extract_to = ''

# Check if any of the folders already exist
if (os.path.exists('raw/') or \
    os.path.exists('interim/') or \
    os.path.exists('processed/')):

    print(f"Some data apears to already exist! To replace the data folders, please first delete the 'raw', 'interim' and 'processed' folders")

else:
    # If archive file doesn't exist, download from S3
    if os.path.exists(path_to_zip_file)==False:
        print(f'Downloading {path_to_zip_file} (approx. 500 MB)...', end=' ')
        url = 'https://ojd-mapping-career-causeways.s3.eu-west-2.amazonaws.com/data.zip'
        urllib.request.urlretrieve(url, path_to_zip_file)
        print('Done!')

    # Extract the archive
    print(f'Extracting the archive in {directory_to_extract_to}...', end=' ')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print('Done!')
