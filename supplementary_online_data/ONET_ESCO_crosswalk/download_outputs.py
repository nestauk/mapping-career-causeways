import urllib.request
import os
import zipfile

fpath = ''
path_to_zip_file = fpath + 'outputs.zip'
directory_to_extract_to = 'outputs/'
if os.path.exists(directory_to_extract_to)==False:
    os.mkdir(directory_to_extract_to)

if os.path.exists(path_to_zip_file)==False:
    print(f'Downloading {path_to_zip_file} (264.7 MB)...', end=' ')
    url = 'https://ojd-mapping-career-causeways.s3.eu-west-2.amazonaws.com/outputs.zip'
    urllib.request.urlretrieve(url, path_to_zip_file)
    print('Done!')

print(f'Extracting the archive in {directory_to_extract_to}...', end=' ')
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
print('Done!')
