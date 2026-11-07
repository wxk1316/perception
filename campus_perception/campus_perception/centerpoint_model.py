"""Fixed Autoware CenterPoint base/tiny ONNX inference without torch/spconv.

NumPy performs XYZT pillar construction and decoding. TensorRT 10 and libcudart
perform inference on Orin; ONNX Runtime is an optional offline validation backend.
This module does not depend on Autoware ROS packages or change model weights.
"""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

from .campus_cloud import transform_points
from .temporal_cloud import rectangle_distance


class CudaRuntime:
    def __init__(self):
        candidates = [ctypes.util.find_library('cudart'), 'libcudart.so.12',
                      '/usr/local/cuda/lib64/libcudart.so',
                      '/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so']
        self.lib = None
        for name in candidates:
            if not name:
                continue
            try:
                self.lib = ctypes.CDLL(name);break
            except OSError:
                pass
        if self.lib is None:
            raise RuntimeError('CUDA runtime not found; run this backend on the Orin vehicle')
        signatures = {
            'cudaMalloc':[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t],
            'cudaFree':[ctypes.c_void_p],
            'cudaMemcpy':[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int],
            'cudaStreamCreate':[ctypes.POINTER(ctypes.c_void_p)],
            'cudaStreamDestroy':[ctypes.c_void_p],
            'cudaStreamSynchronize':[ctypes.c_void_p],
            'cudaGetDevice':[ctypes.POINTER(ctypes.c_int)],
            'cudaDeviceGetAttribute':[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int],
        }
        for name,args in signatures.items():
            function = getattr(self.lib,name);function.argtypes=args;function.restype=ctypes.c_int
        self.stream = ctypes.c_void_p()
        self.call('cudaStreamCreate',ctypes.byref(self.stream))

    def call(self,name,*args):
        error = getattr(self.lib,name)(*args)
        if error:
            raise RuntimeError(f'{name} failed with CUDA error {error}')

    def architecture(self):
        device,major,minor = ctypes.c_int(),ctypes.c_int(),ctypes.c_int()
        self.call('cudaGetDevice',ctypes.byref(device))
        self.call('cudaDeviceGetAttribute',ctypes.byref(major),75,device.value)
        self.call('cudaDeviceGetAttribute',ctypes.byref(minor),76,device.value)
        return f'sm{major.value}{minor.value}'


class TensorRTGraph:
    def __init__(self,path,allow_build=False,encoder=False,input_shape=None):
        import tensorrt as trt
        if int(trt.__version__.split('.')[0]) != 10:
            raise RuntimeError('Campus TensorRT backend requires TensorRT 10.x')
        self.trt,self.cuda = trt,CudaRuntime()
        self.logger = trt.Logger(trt.Logger.WARNING)
        path = Path(path)
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        shape_key = 'N40000x32x9' if encoder else 'x'.join(map(str,input_shape or ()))
        cache = path.with_name(path.stem+f'.{fingerprint}.{shape_key}.trt{trt.__version__}.{self.cuda.architecture()}.{platform.machine()}.engine')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(cache.read_bytes()) if cache.exists() else None
        if self.engine is None:
            if not allow_build:
                raise RuntimeError('Engine missing/incompatible. Run campus_model --build-only --model-directory '+str(path.parent))
            builder = trt.Builder(self.logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network,self.logger)
            if not parser.parse(path.read_bytes()):
                raise RuntimeError('ONNX parse failed: '+'; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1024*1024*1024)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if any(size<0 for size in network.get_input(0).shape):
                profile = builder.create_optimization_profile()
                if encoder:
                    profile.set_shape(network.get_input(0).name,(1,32,9),(4000,32,9),(40000,32,9))
                elif input_shape is not None:
                    profile.set_shape(network.get_input(0).name,input_shape,input_shape,input_shape)
                else:
                    raise RuntimeError('Dynamic ONNX graph requires an explicit input profile')
                config.add_optimization_profile(profile)
            serialized = builder.build_serialized_network(network,config)
            if serialized is None:
                raise RuntimeError('TensorRT engine build failed')
            temporary = cache.with_suffix('.engine.tmp')
            temporary.write_bytes(bytes(serialized));temporary.replace(cache)
            self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError('TensorRT engine could not be loaded')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT execution context could not be created')
        self.inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                       if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.outputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                        if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        self.buffers = {}

    def infer(self,array):
        if len(self.inputs)!=1:
            raise RuntimeError('Unexpected model input count')
        input_name = self.inputs[0]
        dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(input_name)))
        array = np.ascontiguousarray(array,dtype=dtype)
        if not self.context.set_input_shape(input_name,array.shape):
            raise RuntimeError('Invalid TensorRT input shape')
        host = {input_name:array}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(size<=0 for size in shape):
                raise RuntimeError('Unresolved TensorRT output shape')
            host[name] = np.empty(shape,dtype=self.trt.nptype(self.engine.get_tensor_dtype(name)))
        for name,value in host.items():
            old = self.buffers.get(name)
            if old is None or old[1]<value.nbytes:
                if old:
                    self.cuda.call('cudaFree',old[0])
                pointer = ctypes.c_void_p()
                self.cuda.call('cudaMalloc',ctypes.byref(pointer),value.nbytes)
                self.buffers[name] = (pointer,value.nbytes)
            if not self.context.set_tensor_address(name,self.buffers[name][0].value):
                raise RuntimeError('TensorRT tensor binding failed')
        self.cuda.call('cudaMemcpy',self.buffers[input_name][0],ctypes.c_void_p(array.ctypes.data),array.nbytes,1)
        if not self.context.execute_async_v3(self.cuda.stream.value):
            raise RuntimeError('TensorRT inference failed')
        self.cuda.call('cudaStreamSynchronize',self.cuda.stream)
        for name in self.outputs:
            value = host[name]
            self.cuda.call('cudaMemcpy',ctypes.c_void_p(value.ctypes.data),self.buffers[name][0],value.nbytes,2)
        return {name:host[name] for name in self.outputs}

    def close(self):
        for pointer,_ in self.buffers.values():
            self.cuda.call('cudaFree',pointer)
        self.buffers.clear()
        self.cuda.call('cudaStreamDestroy',self.cuda.stream)


class OnnxGraph:
    def __init__(self,path,**_):
        import onnxruntime as ort
        options = ort.SessionOptions();options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def infer(self,array):
        values = self.session.run(None,{self.session.get_inputs()[0].name:np.asarray(array,dtype=np.float32)})
        return {tensor.name:value for tensor,value in zip(self.session.get_outputs(),values)}

    def close(self):
        pass


def pillar_features(points,params):
    """Construct the model's actual [N,32,9] XYZT + residual input."""
    points = np.asarray(points,dtype=np.float32).reshape(-1,4)
    lower,upper = np.array(params['point_cloud_range'][:3]),np.array(params['point_cloud_range'][3:])
    voxel = np.asarray(params['voxel_size'])
    keep = np.isfinite(points).all(axis=1)&np.all(points[:,:3]>=lower,axis=1)&np.all(points[:,:3]<upper,axis=1)
    points = points[keep]
    coords = np.floor((points[:,:2]-lower[:2])/voxel[:2]).astype(np.int32)
    grid = np.rint((upper[:2]-lower[:2])/voxel[:2]).astype(int)
    if not len(points):
        return np.zeros((1,32,9),dtype=np.float32),np.empty((0,2),dtype=int)
    linear = coords[:,1]*grid[0]+coords[:,0]
    order = np.argsort(linear,kind='stable')
    keys,starts,counts = np.unique(linear[order],return_index=True,return_counts=True)
    selected = np.arange(len(keys))
    maximum = int(params['max_voxel_size'])
    if len(keys)>maximum:
        centres = np.column_stack((keys%grid[0],keys//grid[0]))*voxel[:2]+lower[:2]+voxel[:2]/2
        selected = np.argsort(np.linalg.norm(centres,axis=1),kind='stable')[:maximum]
    keys,starts,counts = keys[selected],starts[selected],counts[selected]
    n = len(keys)
    features = np.zeros((n,32,9),dtype=np.float32)
    ranks = np.arange(32)[None,:]
    valid = ranks<counts[:,None]
    # Uniform samples preserve several beams/history times in dense pillars.
    offsets = np.minimum((ranks*counts[:,None]//np.minimum(counts,32)[:,None]),counts[:,None]-1)
    samples = points[order[starts[:,None]+offsets]]
    samples[~valid] = 0
    features[:,:,:4] = samples
    means = samples[:,:,:3].sum(axis=1)/np.minimum(counts,32)[:,None]
    features[:,:,4:7] = samples[:,:,:3]-means[:,None,:]
    xy = np.column_stack((keys%grid[0],keys//grid[0]))
    centres = xy*voxel[:2]+lower[:2]+voxel[:2]/2
    features[:,:,7:9] = samples[:,:,:2]-centres[:,None,:]
    features[~valid] = 0
    return features,xy


def box_iou(a,b):
    """BEV AABB overlap, used only for duplicate suppression of model regions."""
    bounds = []
    for box in (a,b):
        c,s = abs(np.cos(box['yaw'])),abs(np.sin(box['yaw']))
        extent = np.array([c*box['length']+s*box['width'],s*box['length']+c*box['width']])
        centre = np.array([box['x'],box['y']])
        bounds.append((centre-extent/2,centre+extent/2,np.prod(extent)))
    overlap = np.prod(np.maximum(np.minimum(bounds[0][1],bounds[1][1])-np.maximum(bounds[0][0],bounds[1][0]),0))
    return float(overlap/(bounds[0][2]+bounds[1][2]-overlap+1e-9))


def decode_predictions(outputs,params,stamp,world_transform,score_threshold=.35,max_range=5.):
    """Decode every directly trained label of the fixed five-class head."""
    expected = ('heatmap','reg','height','dim','rot','vel')
    if any(name not in outputs for name in expected):
        raise RuntimeError('Unexpected CenterPoint output names')
    arrays = {name:np.asarray(outputs[name])[0] for name in expected}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('Nonfinite CenterPoint output')
    heat = arrays['heatmap']
    if heat.shape[0] != len(params['class_names']):
        raise RuntimeError('Head class count does not match model manifest')
    labels = np.argmax(heat,axis=0)
    scores = 1/(1+np.exp(-np.clip(np.max(heat,axis=0),-40,40)))
    yi,xi = np.where(scores>=score_threshold)
    if not len(xi):
        return []
    order = np.argsort(scores[yi,xi])[::-1][:300]
    yi,xi = yi[order],xi[order]
    factor = params['downsample_factor'];voxel = params['voxel_size'];lower = params['point_cloud_range']
    boxes = []
    for row,column in zip(yi,xi):
        kind = params['class_names'][labels[row,column]]
        x = (column+arrays['reg'][0,row,column])*voxel[0]*factor+lower[0]
        y = (row+arrays['reg'][1,row,column])*voxel[1]*factor+lower[1]
        width,length,height = np.exp(np.clip(arrays['dim'][:,row,column],-6,4))
        if not (.1<=width<=10 and .1<=length<=20 and .1<=height<=8):
            continue
        sine,cosine = arrays['rot'][:2,row,column]
        if np.hypot(sine,cosine)<params.get('yaw_norm_thresholds',[0]*5)[labels[row,column]]:
            continue
        yaw = float(np.arctan2(sine,cosine))
        if rectangle_distance(x,y,float(length),float(width),yaw=yaw) > max_range:
            continue
        centre = transform_points(np.array([[x,y,arrays['height'][0,row,column]]]),world_transform)[0]
        # Project the box axes through the calibrated sensor rotation.
        axis = world_transform[:3,:3]@np.array([np.cos(yaw),np.sin(yaw),0])
        world_yaw = float(np.arctan2(axis[1],axis[0]))
        # Source attitude can tilt the 3D box footprint on ramps. Project an
        # envelope including height; exact z-centre conventions then remain
        # conservative for current-point region assignment.
        if abs(world_transform[2,0])+abs(world_transform[2,1])>1e-6:
            offsets = np.array([[sx*length/2,sy*width/2,sz*height]
                                for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
            cy,sy = np.cos(yaw),np.sin(yaw)
            offsets = offsets@np.array([[cy,sy,0],[-sy,cy,0],[0,0,1]])
            projected = offsets@world_transform[:3,:3].T
            cw,sw = np.cos(world_yaw),np.sin(world_yaw)
            length = 2*np.max(np.abs(cw*projected[:,0]+sw*projected[:,1]))
            width = 2*np.max(np.abs(-sw*projected[:,0]+cw*projected[:,1]))
        box = {'x':float(centre[0]),'y':float(centre[1]),'yaw':world_yaw,
               'length':float(length),'width':float(width),'class':kind.lower(),
               'score':float(scores[row,column]),'stamp':float(stamp)}
        # Class-aware centre + AABB suppression; geometry remains independent.
        duplicate = False
        for other in boxes:
            if other['class']!=box['class']:
                continue
            radius = .35 if box['class']=='pedestrian' else .5
            iou_limit = .55 if box['class']=='pedestrian' else .3
            if np.hypot(box['x']-other['x'],box['y']-other['y'])<radius or box_iou(box,other)>iou_limit:
                duplicate = True;break
        if not duplicate:
            boxes.append(box)
    return boxes


class CenterPointModel:
    def __init__(self,directory,backend='tensorrt',allow_build=False,score_threshold=.35,max_range=5.):
        import yaml
        directory = Path(directory).expanduser()
        lock = json.loads((directory/'bundle.lock.json').read_text())
        for artifact in lock['files']:
            if hashlib.sha256((directory/artifact['file']).read_bytes()).hexdigest()!=artifact['sha256']:
                raise RuntimeError('Model bundle checksum mismatch: '+artifact['file'])
        manifest = yaml.safe_load((directory/'ml_package.param.yaml').read_text())['/**']['ros__parameters']
        self.params = manifest['model_params']
        if (manifest['version']!='v4.1' or self.params['point_feature_size']!=4 or
            self.params['encoder_in_feature_size']!=9 or self.params['has_variance'] or
            self.params['class_names']!=['CAR','TRUCK','BUS','BICYCLE','PEDESTRIAN']):
            raise RuntimeError('Only the fixed v4.1 base/tiny XYZT bundle is supported')
        graph = TensorRTGraph if backend=='tensorrt' else OnnxGraph if backend=='onnxruntime' else None
        if graph is None:
            raise ValueError('Unsupported model backend')
        lo,hi = np.array(self.params['point_cloud_range'][:2]),np.array(self.params['point_cloud_range'][3:5])
        grid = np.rint((hi-lo)/np.array(self.params['voxel_size'][:2])).astype(int)
        self.spatial = np.zeros((1,32,int(grid[1]),int(grid[0])),dtype=np.float32)
        self.encoder = graph(directory/manifest['encoder_onnx_path'],allow_build=allow_build,encoder=True)
        self.head = graph(directory/manifest['head_onnx_path'],allow_build=allow_build,input_shape=self.spatial.shape)
        self.score_threshold,self.max_range = float(score_threshold),float(max_range)

    def infer(self,xyz,stamp,world_transform):
        points = np.asarray(xyz,dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (3,4):
            raise ValueError('Model expects XYZ or externally fused XYZT')
        if points.shape[1] == 3:
            points = np.column_stack((points,np.zeros(len(points),dtype=np.float32)))
        if not np.isfinite(points).all() or np.any(points[:,3]<0):
            raise ValueError('Invalid source age or model points')
        # The ROS source cache provides the latest two frames. Never fuse again
        # here: inference may skip raw frames while busy.
        features,coords = pillar_features(points,self.params)
        pillar = self.encoder.infer(features)['pillar_features'].reshape(-1,32)
        self.spatial.fill(0)
        if len(coords):
            self.spatial.reshape(32,-1)[:,coords[:,1]*self.spatial.shape[3]+coords[:,0]] = pillar[:len(coords)].T
        outputs = self.head.infer(self.spatial)
        return decode_predictions(outputs,self.params,stamp,world_transform,self.score_threshold,self.max_range)

    def close(self):
        self.encoder.close();self.head.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory',type=Path,required=True)
    parser.add_argument('--backend',choices=('tensorrt','onnxruntime'),default='tensorrt')
    parser.add_argument('--build-only',action='store_true')
    args = parser.parse_args()
    model = CenterPointModel(args.model_directory,args.backend,allow_build=args.build_only)
    print('Loaded CenterPoint',args.backend,'with',model.params['class_names'])
    model.close()


if __name__=='__main__':
    main()
