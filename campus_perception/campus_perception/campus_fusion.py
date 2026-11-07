"""Source-time neural entity IDs plus independent current geometric coverage."""
from .frame_utils import oriented_box_aabb

MODEL_ID_OFFSET = 1_000_000_000


def model_detections(boxes):
    result = []
    for box in boxes:
        length,width = oriented_box_aabb(box['length'],box['width'],box['yaw'])
        result.append((box['x'],box['y'],box['class'],length,width,box['score']))
    return result


def fuse_coverage(geometry,entities):
    """Drop only geometric rectangles completely enclosed by a model envelope.

    Model snapshots already include source-age reachable bounds. Geometry outside
    them stays published, including fragments of partially underestimated boxes.
    """
    models = [(MODEL_ID_OFFSET+item[0],*item[1:]) for item in entities]
    retained = []
    for item in geometry:
        covered = any(abs(item[1]-model[1])+item[6]/2<=model[6]/2+1e-9 and
                      abs(item[2]-model[2])+item[7]/2<=model[7]/2+1e-9 for model in models)
        if not covered:
            retained.append(item)
    return retained+models


def expand_publication_delay(tracks,delay,measurement_ages,speed_limits,velocity_max_age=.25):
    """Keep the source header while covering the additional delivery delay."""
    delay = max(float(delay),0.)
    result = []
    for item in tracks:
        tid,x,y,speed,heading,kind,length,width = item
        rate = float(speed_limits.get(kind,max(speed_limits.values())))+max(float(speed),0.)+.2
        growth = 2*rate*delay
        if measurement_ages.get(tid,0.)+delay>velocity_max_age:
            kind,speed,heading = 'unknown',0.,0.
        result.append((tid,x,y,speed,heading,kind,length+growth,width+growth))
    return result
