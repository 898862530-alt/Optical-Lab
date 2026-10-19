import io
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from app import app, cache
from depth_engine import DepthField, align_prediction, network_size
from optics import render_original, render_plan, make_plan
from test_optics import P


def finished(client,response):
    assert response.status_code==202,response.get_json()
    token=response.get_json()['job']
    for _ in range(500):
        job=client.get('/api/jobs/'+token).get_json()
        if job['state']=='done':return job['result']
        if job['state']=='error':raise AssertionError(job['message'])
        time.sleep(.01)
    raise AssertionError('job did not complete')


class ResolutionTests(unittest.TestCase):
    def test_official_preprocessing(self):
        self.assertEqual(network_size(1500,1000,518),(784,518))
        w,h=network_size(8000,1000,1008)
        self.assertEqual(w%14,0);self.assertEqual(h%14,0)
        self.assertLess(w*h,1_500_000)

    def test_affine_alignment(self):
        x=np.linspace(0,1,400,dtype=np.float32).reshape(20,20)
        np.testing.assert_allclose(align_prediction(x,3*x+7),3*x+7,atol=1e-5)

    def test_float_precision_and_tile_coordinates(self):
        rgb=np.full((120,180,3),120,np.uint8)
        d=np.tile(np.linspace(0,1,90,dtype=np.float32),(60,1))
        field=DepthField(rgb,d)
        full=field.region(rgb,0,0,180,120)
        part=field.region(rgb,50,30,80,90)
        np.testing.assert_array_equal(part,full[30:90,50:80])
        self.assertEqual(full.dtype,np.float32)
        self.assertGreater(np.unique(full).size,100)

    def test_no_tile_seams(self):
        rng=np.random.default_rng(9)
        rgb=rng.integers(0,256,(151,223,3),dtype=np.uint8)
        q=np.tile(np.linspace(0,1,223,dtype=np.float32),(151,1))
        field=DepthField(rgb,q)
        tiled,stats=render_original(rgb,field,P,tile_size=48)
        focus=field.sample(rgb,.5,.5)
        plan=make_plan(223,151,P,focus)
        whole=render_plan(rgb,field.region(rgb,0,0,223,151),plan)
        self.assertLessEqual(np.abs(tiled.astype(int)-whole.astype(int)).max(),1)

    def test_original_pixels_and_export_dimensions(self):
        # Large enough to reveal a hidden 1200 px reduction, with non-reconstructable pixels.
        rgb=np.random.default_rng(5).integers(0,256,(1234,1801,3),dtype=np.uint8)
        def estimate(original,progress):
            field=DepthField(original,np.full((240,350),.5,np.float32))
            return field,dict(model='test',passes=3,input_sizes=[[728,518],[1400,1008],[1400,1008]])
        c=app.test_client();buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG');buffer.seek(0)
        with patch('app.estimate',estimate):
            d=finished(c,c.post('/api/analyze',data={'photo':(buffer,'original.png')}))
        self.assertEqual((d['width'],d['height']),(1801,1234))
        self.assertEqual(d['preview_width'],1200)
        np.testing.assert_array_equal(cache[d['id']]['rgb'],rgb)
        p=dict(P,id=d['id'],confirmed=True,target_aperture=4)
        out=finished(c,c.post('/api/export-original',json=p))
        response=c.get(out['download'])
        self.assertIn('attachment',response.headers['Content-Disposition'])
        im=Image.open(io.BytesIO(response.data));self.assertEqual(im.size,(1801,1234))
        np.testing.assert_array_equal(np.asarray(im),rgb)
        self.assertIn('icc_profile',im.info)
        dep=finished(c,c.post('/api/export-depth',json={'id':d['id']}))
        depth_image=Image.open(io.BytesIO(c.get(dep['download']).data))
        self.assertEqual(depth_image.size,(1801,1234))
        self.assertEqual(depth_image.mode,'I;16')

    def test_exif_orientation_original_dimensions(self):
        rgb=np.full((48,72,3),128,np.uint8)
        exif=Image.Exif();exif[274]=6
        f=io.BytesIO();Image.fromarray(rgb).save(f,format='JPEG',exif=exif);f.seek(0)
        def estimate(rgb,progress):
            return DepthField(rgb,np.full((24,36),.5,np.float32)),{}
        c=app.test_client()
        with patch('app.estimate',estimate):
            d=finished(c,c.post('/api/analyze',data={'photo':(f,'rotated.jpg')}))
        self.assertEqual((d['width'],d['height']),(48,72))

class InputIntegrityTests(unittest.TestCase):
    def test_reject_rgb16_png(self):
        import struct,zlib
        def chunk(name,data):
            return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data))
        data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',2,2,16,2,0,0,0))
        data+=chunk(b'IDAT',zlib.compress((b'\x00'+b'\xff'*12)*2))+chunk(b'IEND',b'')
        response=app.test_client().post('/api/analyze',data={'photo':(io.BytesIO(data),'16bit.png')})
        self.assertEqual(response.status_code,400)

    def test_preserve_alpha_channel(self):
        from app import png,read_photo
        from werkzeug.datastructures import FileStorage
        rgb=np.full((16,24,3),119,np.uint8)
        alpha=np.tile(np.arange(24,dtype=np.uint8)*10,(16,1))
        im=Image.fromarray(rgb);im.putalpha(Image.fromarray(alpha))
        data=io.BytesIO();im.save(data,format='PNG');data.seek(0)
        original,a,meta=read_photo(FileStorage(stream=data,filename='alpha.png'))
        np.testing.assert_array_equal(a,alpha)
        out=Image.open(io.BytesIO(png(original,a)))
        np.testing.assert_array_equal(np.asarray(out.getchannel('A')),alpha)

    def test_export_filename(self):
        from app import save_export
        c=app.test_client()
        response=c.get(save_export(b'PNG','depth16.png'))
        self.assertIn('depth16.png',response.headers['Content-Disposition'])

if __name__=='__main__':unittest.main()
