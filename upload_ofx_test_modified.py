from src.main import app
import io

client = app.test_client()
path = r"mariaconciliadora-backend/samples/2021_teste_discrepancias.ofx"
with open(path, 'rb') as f:
    orig = f.read()
# append a newline to change file hash
payload = io.BytesIO(orig + b"\n")

data = {
    'file': (payload, '2021_teste_discrepancias_modified.ofx')
}
resp = client.post('/api/upload-ofx', data=data, content_type='multipart/form-data')
print('STATUS', resp.status_code)
try:
    print('JSON', resp.get_json())
except Exception as e:
    print('RAW', resp.data[:500])
