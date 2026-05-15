import requests

def upload_ec2(pred, filename):
    # Get pre-signed URL from EC2
    # Change to current IP
    resp = requests.get("http://172.31.21.21:5000/get_upload_url", params={'filename': filename, 'prediction': pred}).json()
    upload_url = resp['url']

    # Upload file using the URL
    with open(filename,'rb') as f:
        r = requests.put(upload_url, data=f)

    print("[{}]: {} UPLOADED".format(
        r.status_code,
        filename
        ))

def download_ec2(filename):
    resp = requests.get(
        # Change to current IP
        "http://172.31.21.21:5000/get_download_url",
        params = {'key': filename}
        ).json()

    download_url = resp['url']

    r = requests.get(download_url)

    with open(filename, 'wb') as f:
        f.write(r.content)

