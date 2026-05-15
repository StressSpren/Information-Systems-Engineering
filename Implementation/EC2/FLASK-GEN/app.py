from flask import Flask, jsonify, request
import boto3

app = Flask(__name__)
s3 = boto3.client('s3', region_name='eu-north-1')
BUCKET = 'flump-ocr-bucket'

def generate_upload_url(key, expiry=600):
    return s3.generate_presigned_url(
        'put_object',
        Params={'Bucket': BUCKET, 'Key': key},
        ExpiresIn=expiry
    )

def generate_download_url(key, expiry=600):
    return s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET, 'Key': key},
        ExpiresIn=expiry
    )

@app.route('/get_upload_url', methods=['GET'])
def get_upload_url():
    filename = request.args.get('filename')
    url = generate_upload_url(f"uploads/{filename}")
    return jsonify({'url': url, 'filename': filename})

@app.route('/get_download_url', methods=['GET'])
def get_download_url():
    key = request.args.get('key')
    url = generate_download_url(key)
    return jsonify({'url': url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
