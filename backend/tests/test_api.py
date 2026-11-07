from fastapi.testclient import TestClient
from app.main import app

def test_endpoints():
    with TestClient(app) as c:
        h=c.get('/api/health')
        assert h.status_code==200
        assert h.json()['labels']==100
        l=c.get('/api/labels')
        assert len(l.json()['labels'])==100
        t=c.post('/api/translate',json={'glosses':['Tôi','Muốn','Uống','Nước']})
        assert t.status_code==200
        assert 'sentence' in t.json()
        bad=c.post('/api/predict/video',files={'file':('bad.txt',b'hello','text/plain')})
        assert bad.status_code==415
