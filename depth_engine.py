"""FP32 Large, aspect-preserving multi-scale / flip inference and guided reconstruction."""
from pathlib import Path
import math
import threading
import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
_MODEL_LOCK = threading.Lock()
_SESSION = None


def network_size(width, height, short_side, max_pixels=1_450_000):
    # Follow official lower-bound preprocessing, with a disclosed memory bound.
    scale = short_side / min(width, height)
    if width * height * scale * scale > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
    return max(14, round(width * scale / 14) * 14), max(14, round(height * scale / 14) * 14)


def align_prediction(source, reference):
    """Fit positive affine scale in relative inverse depth, reject extreme residuals."""
    x, y = source[::4, ::4].ravel(), reference[::4, ::4].ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 10 or float(np.var(x)) < 1e-10:
        return source.copy()
    a, b = 1., 0.
    for _ in range(3):
        xv, yv = x, y
        a = float(np.mean((xv-xv.mean())*(yv-yv.mean())) / max(float(np.var(xv)),1e-10))
        b = float(yv.mean() - a*xv.mean())
        residual = np.abs(a*x+b-y)
        keep = residual <= max(float(np.percentile(residual,90)),1e-6)
        if keep.sum() < 10: break
        x,y=x[keep],y[keep]
    if a <= 0 or not np.isfinite(a+b):
        raise ValueError('深度预测之间无法可靠对齐，请重新分析照片。')
    return source * a + b


def gray(rgb):
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255


def guided_coefficients(rgb, depth, radius=5, epsilon=.0025):
    """Scalar guided filter. Regularization limits texture transfer from RGB."""
    guide = gray(cv2.resize(rgb,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_AREA))
    size=(radius*2+1,)*2
    mean=lambda x:cv2.boxFilter(x,-1,size,borderType=cv2.BORDER_REFLECT)
    mi,mp=mean(guide),mean(depth)
    covariance=mean(guide*depth)-mi*mp
    variance=np.maximum(mean(guide*guide)-mi*mi,0)
    a=covariance/(variance+epsilon)
    b=mp-a*mi
    return mean(a),mean(b)


class DepthField:
    """Float32 field; reconstruct tiles at original coordinates without an 8-bit round trip."""
    def __init__(self,rgb,normalized):
        self.height,self.width=rgb.shape[:2]
        self.coarse=normalized.astype(np.float32)
        self.a,self.b=guided_coefficients(rgb,self.coarse)

    def region(self,rgb,x0,y0,x1,y1):
        ch,cw=self.coarse.shape
        xs=((np.arange(x0,x1,dtype=np.float32)+.5)*cw/self.width-.5)
        ys=((np.arange(y0,y1,dtype=np.float32)+.5)*ch/self.height-.5)
        mx,my=np.meshgrid(xs,ys)
        resize=lambda a:cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        guided=resize(self.a)*gray(rgb[y0:y1,x0:x1])+resize(self.b)
        # Keep high frequency geometry predicted by the network; restrained edge guidance.
        return np.clip(.7*resize(self.coarse)+.3*guided,0,1).astype(np.float32)

    def sample(self,rgb,x,y):
        px,py=round(x*(self.width-1)),round(y*(self.height-1))
        return float(np.median(self.region(rgb,max(0,px-2),max(0,py-2),min(self.width,px+3),min(self.height,py+3))))

    def preview(self,rgb,width,height):
        small=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
        d=cv2.resize(self.coarse,(width,height),interpolation=cv2.INTER_LINEAR)
        a=cv2.resize(self.a,(width,height),interpolation=cv2.INTER_LINEAR)
        b=cv2.resize(self.b,(width,height),interpolation=cv2.INTER_LINEAR)
        return np.clip(.7*d+.3*(a*gray(small)+b),0,1).astype(np.float32)


def estimate(rgb,progress=lambda *args:None):
    global _SESSION
    path=ROOT/'models/depth-large.onnx'
    if not path.exists():
        raise ValueError('Large 模型缺失，请运行 bash setup.sh 安装高精度模型。')
    with _MODEL_LOCK:
        if _SESSION is None:
            progress(.03,'正在加载 Large FP32 模型…')
            opts=ort.SessionOptions()
            opts.intra_op_num_threads=4
            opts.inter_op_num_threads=1
            opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            opts.enable_cpu_mem_arena=False
            opts.enable_mem_pattern=False
            _SESSION=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        h,w=rgb.shape[:2]
        # Inference sees original RGB, never the 1200 px preview.
        scale=min(1,2048/max(h,w)); outw,outh=max(1,round(w*scale)),max(1,round(h*scale))
        predictions=[]; sizes=[]
        for i,(short,flip) in enumerate(((518,False),(1008,False),(1008,True))):
            nw,nh=network_size(w,h,short)
            progress(.1+i*.25,f'高精度深度 {i+1}/3 · {nw} × {nh}'+(' · 镜像一致性' if flip else ''))
            image=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_CUBIC)
            if flip: image=np.ascontiguousarray(image[:,::-1])
            value=image.astype(np.float32)/255
            value=(value-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
            tensor=np.ascontiguousarray(value.transpose(2,0,1)[None])
            raw=_SESSION.run(None,{_SESSION.get_inputs()[0].name:tensor})[0].squeeze()
            if flip:raw=raw[:,::-1]
            predictions.append(cv2.resize(raw,(outw,outh),interpolation=cv2.INTER_CUBIC))
            sizes.append([nw,nh])
            del tensor,value,image,raw
        progress(.9,'正在融合尺度与细化深度边缘…')
        # High resolution is the reference; low resolution contributes global layout only.
        high=predictions[1]
        low=align_prediction(predictions[0],high)
        flipped=align_prediction(predictions[2],high)
        fused=(high+flipped)*.5
        # Low-frequency context from the smaller scale, retaining fine structure of high-res.
        sigma=max(3,min(outh,outw)/80)
        fused += .15*(cv2.GaussianBlur(low,(0,0),sigma)-cv2.GaussianBlur(fused,(0,0),sigma))
        lo,hi=np.percentile(fused,[.1,99.9])
        normalized=np.clip((fused-lo)/max(float(hi-lo),1e-6),0,1).astype(np.float32)
        info=dict(model='Depth Anything V2 Large · FP32',passes=3,input_sizes=sizes,field_size=[outw,outh],precision='float32',license='CC-BY-NC-4.0')
        return DepthField(rgb,normalized),info
