"""Model labels stay distinct; only the legacy planner gets collapsed labels."""
MODEL_CLASSES = ('car', 'truck', 'bus', 'bicycle', 'pedestrian')
ACTOR_CLASSES = MODEL_CLASSES + ('vehicle',)  # compatibility with older inputs


def planner_obstacles(items):
    result = []
    for tid, x, y, speed, heading, kind, length, width in items:
        if speed < 0:
            kind, speed, heading = 'unknown', 0., 0.
        elif kind in ('car', 'truck', 'bus'):
            kind = 'vehicle'
        result.append((tid, x, y, speed, heading, kind, length, width))
    return result
