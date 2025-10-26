from src.main import app
from werkzeug.datastructures import FileStorage

client = app.test_client()
path = r"mariaconciliadora-backend/samples/2021_teste_discrepancias.ofx"
with open(path, 'rb') as f:
    data = {
        'file': (f, '2021_teste_discrepancias.ofx')
    }
    resp = client.post('/api/upload-ofx', data=data, content_type='multipart/form-data')
    print('STATUS', resp.status_code)
    try:
        print('JSON', resp.get_json())
    except Exception as e:
        print('RESP', resp.data[:500])
