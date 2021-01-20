import boto3
import os
import sys
import pandas as pd

## USAGE:
# python download_from_s3.py outputs_Essential/

## SETUP

# Set up directory names
if len(sys.argv) < 2:
    print('Provide <s3 folder name> and <local folder> as inputs!')
    raise
elif len(sys.argv) < 3:
    s3_folder = sys.argv[1]
    local_dir = None
else:
    s3_folder = sys.argv[1]
    local_dir = sys.argv[2]

df_keys = pd.read_csv('../../private/karlisKanders_accessKeys.csv')
os.environ["AWS_ACCESS_KEY_ID"] = df_keys['Access key ID'].iloc[0]
os.environ["AWS_SECRET_ACCESS_KEY"] = df_keys['Secret access key'].iloc[0]

bucket_name = 'ojd-temp-storage'
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix = s3_folder):
        if obj.key == s3_folder:
            continue

        if local_dir is None:
            target = obj.key
        else:
            target = os.path.join(local_dir, os.path.basename(obj.key))

        print(os.path.basename(obj.key))

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        bucket.download_file(obj.key, target)

## RUN
download_s3_folder(bucket_name, s3_folder, local_dir)
