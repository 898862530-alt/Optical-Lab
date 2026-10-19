import unittest
import numpy as np
from optics import coc, target_focal, active_diagonal, render
from app import app, metadata
from PIL import Image

P=dict(focal=50,aperture=4,sensor_w=36,sensor_h=24,target_w=36,target_h=24,target_aperture=1.4,near=1.5,far=12,focus_x=.5,focus_y=.5)
class OpticsTests(unittest.TestCase):
    def test_focus_is_zero(self): self.assertEqual(coc(2000,2000,50,4),0)
    def test_aperture_doubles_coc(self): self.assertAlmostEqual(coc(5000,2000,50,2),2*coc(5000,2000,50,4))
    def test_infinity_limit(self): self.assertAlmostEqual(coc(1e15,2000,50,4),2500/(4*1950),places=8)
    def test_matching_magnification(self):
        f=target_focal(50,2000,2)
        self.assertAlmostEqual(f/(2000-f),2*50/(2000-50))
    def test_orientation(self): self.assertEqual(active_diagonal(36,24,900,600),active_diagonal(36,24,600,900))
    def test_identity(self):
        rgb=np.random.default_rng(3).integers(0,256,(48,64,3),dtype=np.uint8)
        d=np.tile(np.linspace(0,1,64,dtype=np.float32),(48,1))
        out,_=render(rgb,d,dict(P,target_aperture=4))
        np.testing.assert_array_equal(out,rgb)
    def test_preserve_constant(self):
        rgb=np.full((48,64,3),137,np.uint8);d=np.tile(np.linspace(0,1,64,dtype=np.float32),(48,1))
        out,_=render(rgb,d,P);self.assertLessEqual(abs(out.astype(float)-137).max(),1)
    def test_reject_smaller_sensor(self):
        with self.assertRaises(ValueError): render(np.zeros((10,10,3),np.uint8),np.zeros((10,10),np.float32),dict(P,target_w=10,target_h=10))
    def test_reject_smaller_aperture(self):
        with self.assertRaises(ValueError): render(np.zeros((10,10,3),np.uint8),np.zeros((10,10),np.float32),dict(P,target_aperture=8))
    def test_metadata_missing(self): self.assertIsNone(metadata(Image.new('RGB',(10,10)))['sensor_w'])
    def test_metadata_exif(self):
        im=Image.new('RGB',(10,10));ex=im.getexif();ex[37386]=50;ex[33437]=2.8;ex[41989]=50
        m=metadata(im);self.assertAlmostEqual(m['sensor_w'],36,places=4);self.assertEqual(m['aperture'],2.8)
    def test_bad_upload(self):
        import io
        c=app.test_client();r=c.post('/api/analyze',data={'photo':(io.BytesIO(b'bad'),'bad.jpg')})
        self.assertEqual(r.status_code,400)
    def test_session_expired(self): self.assertEqual(app.test_client().post('/api/render',json={'id':'bad'}).status_code,400)
if __name__=='__main__': unittest.main()
