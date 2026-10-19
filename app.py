import base64
import io
import secrets
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, request, send_file, send_from_directory
from depth_engine import estimate
from optics import render, render_original, validate

ROOT=Path(__file__).resolve().parent
app=Flask(__name__,static_folder='static')
app.config['MAX_CONTENT_LENGTH']=100*1024*1024
Image.MAX_IMAGE_PIXELS=60_000_000
cache=OrderedDict()
exports=OrderedDict()
jobs=OrderedDict()
lock=threading.RLock()
worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='optical-hq')
SRGB=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def png(rgb,alpha=None):
    im=Image.fromarray(rgb)
    if alpha is not None: im.putalpha(Image.fromarray(alpha))
    out=io.BytesIO(); im.save(out,format='PNG',icc_profile=SRGB)
    return out.getvalue()


def encode(rgb):
    return 'data:image/png;base64,'+base64.b64encode(png(rgb)).decode()


def read_photo(file):
    signature=file.stream.read(32)
    file.stream.seek(0)
    if signature.startswith(b'\x89PNG\r\n\x1a\n') and len(signature)>24 and signature[24]>8:
        raise ValueError('目前仅支持 8 位照片；检测到 16 位 PNG，请先转换，避免隐式降位深。')
    with Image.open(file.stream) as source:
        if source.format not in ('JPEG','PNG','WEBP','TIFF'):
            raise ValueError('请先将 HEIC/RAW 转换为 JPEG、PNG 或 TIFF。')
        bits=getattr(source,'tag_v2',{}).get(258,8)
        bits=max(bits) if isinstance(bits,(list,tuple)) else bits
        if source.mode.startswith(('I','F')) or bits>8:
            raise ValueError('目前支持 8 位照片的原尺寸 PNG 导出；16 位 TIFF/PNG 请先转换为 8 位，避免隐式降位深。')
        if source.width*source.height>60_000_000:
            raise ValueError('当前支持最多 6000 万像素，图片不会被自动缩小。')
        meta=metadata(source)
        profile=source.info.get('icc_profile')
        source=ImageOps.exif_transpose(source)
        alpha=np.asarray(source.convert('RGBA').getchannel('A')).copy() if 'A' in source.getbands() or 'transparency' in source.info else None
        rgb=source.convert('RGB')
        if profile:
            try:
                rgb=ImageCms.profileToProfile(source if source.mode in ('CMYK','LAB') else rgb,ImageCms.ImageCmsProfile(io.BytesIO(profile)),ImageCms.createProfile('sRGB'),outputMode='RGB')
            except (ValueError,ImageCms.PyCMSError) as error:
                raise ValueError('照片色彩配置无效，请先转换到 sRGB 后上传。') from error
        return np.asarray(rgb).copy(),alpha,meta


def begin_job(action):
    token=secrets.token_urlsafe(20)
    with lock:
        active=sum(v['state'] in ('queued','running') for v in jobs.values())
        if active>=4: raise ValueError('处理队列已满，请等待当前任务完成。')
        jobs[token]=dict(state='queued',progress=0,message='等待处理…')
        while len(jobs)>20:
            first=next(iter(jobs))
            if jobs[first]['state'] in ('queued','running'): break
            jobs.pop(first)
    def update(value,message):
        with lock: jobs[token].update(progress=round(value,3),message=message)
    def run():
        with lock: jobs[token]['state']='running'
        try:
            result=action(update)
            with lock: jobs[token].update(state='done',progress=1,message='处理完成',result=result)
        except Exception as error:
            app.logger.exception('Image processing job failed')
            message=str(error) if isinstance(error,ValueError) else '本地处理失败，请查看服务日志后重试。'
            with lock: jobs[token].update(state='error',message=message)
    worker.submit(run)
    return jsonify(job=token),202


def get_item(token):
    with lock:
        item=cache.get(token)
        if item is not None: cache.move_to_end(token)
    if item is None: raise ValueError('照片会话已失效，请重新上传。')
    return item


def parameters():
    p=request.get_json(silent=True)
    if not isinstance(p,dict): raise ValueError('请求参数无效。')
    item=get_item(p.get('id'))
    if p.get('confirmed') is not True: raise ValueError('请先核实原图参数与距离假设。')
    validate(p)
    return p,item


def save_export(data,filename='optical-lab-original.png'):
    token=secrets.token_urlsafe(20)
    with lock:
        exports[token]=(data,filename)
        while len(exports)>3: exports.popitem(last=False)
    return '/api/export/'+token

def metadata(im):
    ex=im.getexif(); tags=dict(ex)
    try: tags.update(ex.get_ifd(34665))
    except (KeyError,TypeError,ValueError): pass
    def number(k):
        try:
            n=float(tags[k]); return n if np.isfinite(n) and n>0 else None
        except (KeyError,TypeError,ValueError,ZeroDivisionError): return None
    f,n,eq=number(37386),number(33437),number(41989)
    if n is None:
        av=number(37378); n=2**(av/2) if av is not None else None
    # 35mm equivalent is a diagonal-FOV proxy, not a measured sensor dimension.
    diag=43.2666153*f/eq if f and eq else None
    return dict(camera=' '.join(str(tags.get(t,'')) for t in (271,272)).strip() or '未记录相机',focal=f,aperture=n,equivalent=eq,
        sensor_w=diag*3/(13**.5) if diag else None,sensor_h=diag*2/(13**.5) if diag else None,
        sensor_source='由实际/35mm 等效焦距推算，假定 3:2；请核实裁切与传感器比例' if diag else 'EXIF 无可靠传感器尺寸，请手动填写',subject_distance=number(37382))

@app.get('/')
def index(): return send_from_directory(ROOT/'static','index.html')

@app.get('/api/health')
def health(): return jsonify(ready=(ROOT/'models/depth-large.onnx').exists(),version='0.2',model='large-fp32',original_export=True)

@app.get('/api/jobs/<token>')
def job_status(token):
    with lock: job=jobs.get(token)
    if job is None: raise ValueError('任务记录已过期，请重新处理。')
    return jsonify(job)

@app.post('/api/analyze')
def analyze():
    file=request.files.get('photo')
    if not file: raise ValueError('请选择 JPEG、PNG、WebP 或 TIFF 照片。')
    rgb,alpha,meta=read_photo(file)
    def action(progress):
        start=time.monotonic()
        field,info=estimate(rgb,progress)
        progress(.96,'正在生成预览，原图完整保留…')
        im=Image.fromarray(rgb); im.thumbnail((1200,1200))
        preview=np.asarray(im).copy(); ph,pw=preview.shape[:2]
        d=field.preview(rgb,pw,ph)
        token=secrets.token_urlsafe(20)
        with lock:
            cache[token]=dict(rgb=rgb,alpha=alpha,field=field,preview=preview,depth=d,info=info)
            while len(cache)>2:cache.popitem(last=False)
        heat=cv2.cvtColor(cv2.applyColorMap(np.uint8(d*255),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
        info['seconds']=round(time.monotonic()-start,1)
        return dict(id=token,image=encode(preview),depth=encode(heat),metadata=meta,width=rgb.shape[1],height=rgb.shape[0],preview_width=pw,preview_height=ph,analysis=info)
    return begin_job(action)

@app.post('/api/render')
def render_api():
    p,item=parameters()
    q=item['field'].sample(item['rgb'],p['focus_x'],p['focus_y'])
    out,stats=render(item['preview'],item['depth'],p,focus_q=q)
    return jsonify(image=encode(out),stats=stats)

@app.post('/api/export-original')
def export_original():
    p,item=parameters()
    def action(progress):
        out,stats=render_original(item['rgb'],item['field'],p,progress)
        progress(.94,'正在编码原尺寸 PNG…')
        url=save_export(png(out,item['alpha']))
        return dict(download=url,width=out.shape[1],height=out.shape[0],stats=stats)
    return begin_job(action)

@app.post('/api/export-depth')
def export_depth():
    p=request.get_json(silent=True) or {}
    item=get_item(p.get('id'))
    def action(progress):
        rgb,field=item['rgb'],item['field'];h,w=rgb.shape[:2]
        out=np.empty((h,w),np.uint16)
        for y in range(0,h,256):
            for x in range(0,w,1024):
                x1,y1=min(w,x+1024),min(h,y+256)
                out[y:y1,x:x1]=np.uint16(np.round(field.region(rgb,x,y,x1,y1)*65535))
            progress(.9*(y+1)/h,f'16 位深度导出 · {min(h,y+256)}/{h} 行')
        data=io.BytesIO(); Image.fromarray(out).save(data,format='PNG')
        return dict(download=save_export(data.getvalue(),'optical-lab-depth-16bit.png'),width=w,height=h)
    return begin_job(action)

@app.get('/api/export/<token>')
def export_image(token):
    with lock:item=exports.get(token)
    if item is None:raise ValueError('导出已过期，请重新计算。')
    data,filename=item
    return send_file(io.BytesIO(data),mimetype='image/png',as_attachment=True,download_name=filename,max_age=0)

@app.after_request
def headers(response):
    if request.path=='/' or request.path.startswith(('/static/','/api/')):
        response.headers['Cache-Control']='no-store'
    return response

@app.errorhandler(ValueError)
def invalid(e):return jsonify(error=str(e)),400
@app.errorhandler(UnidentifiedImageError)
def invalid_image(e):return jsonify(error='无法读取照片，请使用有效的 JPEG、PNG 或 TIFF。'),400
@app.errorhandler(Image.DecompressionBombError)
def too_many_pixels(e):return jsonify(error='图片超过支持的像素上限，请使用不超过 6000 万像素的照片。'),400
@app.errorhandler(413)
def oversized(e):return jsonify(error='文件超过 100 MB，请使用体积更小的无损格式或 JPEG。'),413
@app.errorhandler(500)
def internal(e):return jsonify(error='处理失败，请查看本地服务日志。'),500

if __name__=='__main__':app.run(host='127.0.0.1',port=8765,threaded=True,debug=False)
