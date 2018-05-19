import os
import tinys3
# Simple Library to upload file to s3 made by Smore Inc. 

conn = tinys3.Connection(os.environ["S3_ACCESS_KEY"], os.environ["S3_SECRET_KEY"], tls=True)

f = open('README.md','rb')
conn.upload('README.md',f,'s3.amazonaws.com/hiroga/shenzhen2018/')
# なぜか .s3.amazonaws.com フォルダにアップロードされる
