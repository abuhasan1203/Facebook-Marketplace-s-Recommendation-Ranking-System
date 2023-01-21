import requests
import os
import csv
import boto3

client = boto3.client('s3')

if not os.path.exists('images'):
    os.makedirs('images')

images = []
with open('images.csv') as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    for row in reader:
        row_dict = {}
        row_dict['index'] = row[0]
        row_dict['id'] = row[1]
        row_dict['product_id'] = row[2]
        row_dict['bucket_link'] = row[3]
        row_dict['image_ref'] = row[4]
        row_dict['create_time'] = row[5]
        images.append(row_dict)
images = images [1:]

for image in images:
    bucket, key = image['bucket_link'].split('/',2)[-1].split('/',1)
    bucket = bucket.split('.')[0]
    print(image['bucket_link'])
    print(bucket)
    print(key)
    client.download_file(bucket, key, os.path.join("images", image['id']+".jpg"))
    # img_data = requests.get(image['bucket_link']).content
    # with open(os.path.join("images", image['id']+".jpg"), 'wb') as handler:
    #     handler.write(img_data)