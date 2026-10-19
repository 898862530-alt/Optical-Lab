"""Thin-lens optics and layered, linear-light circular-aperture renderer."""
import math
import cv2
import numpy as np


def active_diagonal(width, height, image_width, image_height):
    long, short = max(width, height), min(width, height)
    ratio = max(image_width, image_height) / min(image_width, image_height)
    a, b = min(long, short * ratio), min(short, long / ratio)
    return math.hypot(a, b)


def coc(z_mm, focus_mm, focal_mm, fnumber):
    return focal_mm ** 2 * np.abs(z_mm - focus_mm) / (fnumber * z_mm * (focus_mm - focal_mm))


def target_focal(focal, focus, scale):
    # Match magnification at the focus plane, including finite-focus extension.
    return scale * focal * focus / (focus + (scale - 1) * focal)


def validate(p):
    for key in ('focal','aperture','sensor_w','sensor_h','target_w','target_h','target_aperture','near','far'):
        if key not in p or not isinstance(p[key], (int,float)) or not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError('参数必须为有限正数：' + key)
    if not 0.1 <= p['target_aperture'] <= p['aperture'] <= 64:
        raise ValueError('目标光圈只能比原图更大（f 值更小），或相等。')
    if p['target_w'] * p['target_h'] < p['sensor_w'] * p['sensor_h'] - 1e-6:
        raise ValueError('目标传感器面积不能小于原图。')
    if not 0.05 <= p['near'] < p['far'] <= 10000:
        raise ValueError('距离范围需满足 0.05 ≤ 近点 < 远点 ≤ 10000 米。')
    if max(p[k] for k in ('sensor_w','sensor_h','target_w','target_h')) > 200 or p['focal'] > 2000:
        raise ValueError('参数超出 demo 支持范围。')
    if p['near'] * 1000 <= p['focal'] * 1.1:
        raise ValueError('最近距离须大于焦距的 1.1 倍。')
    for k in ('focus_x','focus_y'):
        if not isinstance(p.get(k), (int,float)) or not 0 <= p[k] <= 1:
            raise ValueError('请选择有效的原图对焦位置。')


def make_plan(width, height, p, focus_q):
    validate(p)
    d0 = active_diagonal(p['sensor_w'],p['sensor_h'],width,height)
    d1 = active_diagonal(p['target_w'],p['target_h'],width,height)
    if d1 < d0 - 1e-6:
        raise ValueError('裁切到照片比例后，目标有效成像面积更小，请选择更大的传感器。')
    s = 1000 / (1/p['far'] + focus_q*(1/p['near']-1/p['far']))
    f1 = target_focal(p['focal'],s,d1/d0)
    if p['near']*1000 <= f1*1.1:
        raise ValueError('目标焦距接近场景距离；请校准距离或减小传感器。')
    pixels = math.hypot(width,height)
    def radii(q):
        z=1000/(1/p['far']+q*(1/p['near']-1/p['far']))
        c0=coc(z,s,p['focal'],p['aperture'])*pixels/d0
        c1=coc(z,s,f1,p['target_aperture'])*pixels/d1
        return np.sqrt(np.maximum(c1*c1-c0*c0,0))/2,c1
    endpoints,c1=radii(np.array([0.,1.]))
    maximum=float(endpoints.max())
    # Bound physical blur, independently of export pixel dimensions.
    if maximum/math.hypot(width,height)*1500 > 120 or maximum > 1024:
        raise ValueError('计算的散焦过大，请检查场景距离、焦距和光圈。')
    bins=max(64,min(256,math.ceil(maximum*2)))
    centers=(np.arange(bins,dtype=np.float32)+.5)/bins
    centers[min(bins-1,int(focus_q*bins))]=focus_q
    radii_by_bin=radii(centers)[0]
    stats=dict(focus_m=round(s/1000,3),target_focal=round(f1,2),max_radius=round(maximum,2),max_coc=round(float(c1.max()),2),layers=bins)
    return dict(radii=radii_by_bin,bins=bins,maximum=maximum,stats=stats)


def render_plan(rgb, disparity, plan):
    if plan['maximum'] < .35:
        return rgb.copy()
    color = rgb.astype(np.float32)/255
    color = np.where(color<=.04045,color/12.92,((color+.055)/1.055)**2.4)
    canvas = np.zeros_like(color)
    h,w=disparity.shape
    coverage = np.zeros((h,w),np.float32)
    ids = np.minimum((disparity*plan['bins']).astype(np.int16), plan['bins']-1)
    for i in np.unique(ids):
        mask = (ids==i).astype(np.float32)
        r = float(plan['radii'][i])
        if r < .35:
            alpha, premult = mask, color*mask[:,:,None]
        else:
            n = max(1,math.ceil(r))
            yy,xx = np.mgrid[-n:n+1,-n:n+1]
            kernel = np.clip(r+.5-np.sqrt(xx*xx+yy*yy),0,1).astype(np.float32)
            kernel /= kernel.sum()
            alpha = cv2.filter2D(mask,-1,kernel,borderType=cv2.BORDER_REFLECT)
            premult = cv2.filter2D(color*mask[:,:,None],-1,kernel,borderType=cv2.BORDER_REFLECT)
        alpha = np.clip(alpha,0,1)
        canvas *= (1-alpha[:,:,None])
        canvas += premult
        coverage = alpha+coverage*(1-alpha)
    canvas /= np.maximum(coverage[:,:,None],1e-5)
    canvas = np.where(coverage[:,:,None]>1e-5,canvas,color)
    srgb = np.where(canvas<=.0031308,canvas*12.92,1.055*np.maximum(canvas,0)**(1/2.4)-.055)
    return np.uint8(np.clip(srgb*255+.5,0,255))


def render(rgb, disparity, p, focus_q=None):
    h,w=disparity.shape
    if focus_q is None:
        validate(p)
        x,y=round(p['focus_x']*(w-1)),round(p['focus_y']*(h-1))
        focus_q=float(np.median(disparity[max(0,y-2):y+3,max(0,x-2):x+3]))
    plan=make_plan(w,h,p,focus_q)
    return render_plan(rgb,disparity,plan),plan['stats']


def render_original(rgb, field, p, progress=lambda *args:None, tile_size=640):
    """Render actual source pixels with globally identical bins and adequate tile halos."""
    validate(p)
    h,w=rgb.shape[:2]
    q=field.sample(rgb,p['focus_x'],p['focus_y'])
    plan=make_plan(w,h,p,q)
    if plan['maximum']<.35:
        progress(.95,'设备参数相同，保留原图像素…')
        return rgb.copy(),plan['stats']
    halo=math.ceil(plan['maximum'])+3
    out=np.empty_like(rgb)
    total=math.ceil(w/tile_size)*math.ceil(h/tile_size)
    done=0
    for y in range(0,h,tile_size):
        for x in range(0,w,tile_size):
            x1,y1=min(w,x+tile_size),min(h,y+tile_size)
            lx,ly,rx,ry=max(0,x-halo),max(0,y-halo),min(w,x1+halo),min(h,y1+halo)
            d=field.region(rgb,lx,ly,rx,ry)
            tile=render_plan(rgb[ly:ry,lx:rx],d,plan)
            out[y:y1,x:x1]=tile[y-ly:y1-ly,x-lx:x1-lx]
            done+=1
            progress(.05+.85*done/total,f'原尺寸渲染 {done}/{total} · {w} × {h}')
    return out,plan['stats']
