"""Four-point ground-plane projection. Requires a stationary calibrated camera."""
import math


def solve(matrix, values):
    rows = [list(row) + [value] for row, value in zip(matrix, values)]
    for col in range(len(values)):
        pivot = max(range(col, len(rows)), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-9:
            raise ValueError('Calibration points are degenerate; use four widely spaced floor corners.')
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [v / divisor for v in rows[col]]
        for r in range(len(rows)):
            if r != col:
                factor = rows[r][col]
                rows[r] = [a - factor*b for a, b in zip(rows[r], rows[col])]
    return [row[-1] for row in rows]


def calibration(payload):
    points = payload.get('points')
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError('Select exactly four corresponding floor points.')
    width, height = payload.get('width'), payload.get('height')
    if any(type(v) is not int or not 16 <= v <= 8000 for v in (width, height)):
        raise ValueError('Invalid calibration image size.')
    matrix, values = [], []
    cleaned = []
    for p in points:
        if not isinstance(p, list) or len(p) != 4:
            raise ValueError('Each point must contain image u,v and floor x,y.')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in p):
            raise ValueError('Calibration coordinates must be finite numbers.')
        u, v, x, y = p
        if not (0 <= u <= 1 and 0 <= v <= 1 and -5 <= x <= 5 and .1 <= y <= 8):
            raise ValueError('Use normalized image coordinates and floor coordinates within the map.')
        matrix.extend([[u,v,1,0,0,0,-x*u,-x*v], [0,0,0,u,v,1,-y*u,-y*v]])
        values.extend([x,y]); cleaned.append([u,v,x,y])
    # Reject crossed/concave and nearly collinear quadrilaterals in either space.
    for offset in (0,2):
        poly = [(p[offset],p[offset+1]) for p in cleaned]
        crosses = []
        for i in range(4):
            a,b,c = poly[i],poly[(i+1)%4],poly[(i+2)%4]
            crosses.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
        if not (all(c > 1e-5 for c in crosses) or all(c < -1e-5 for c in crosses)):
            raise ValueError('Click corners around the floor rectangle in order; avoid crossed or collinear points.')
    return {'points': cleaned, 'width': width, 'height': height, 'h': solve(matrix, values)}


def project(box, width, height, config):
    if abs(width / height - config['width'] / config['height']) > .01:
        return None
    x1,y1,x2,y2 = box
    # A box cropped at the bottom does not establish a ground contact point.
    if y2 >= height - 2:
        return None
    u,v = (x1+x2)/2/width, y2/height
    # Restrict estimates to the calibrated quadrilateral; no extrapolation.
    poly = [(p[0], p[1]) for p in config['points']]
    crosses = [(b[0]-a[0])*(v-a[1])-(b[1]-a[1])*(u-a[0])
               for a,b in zip(poly, poly[1:]+poly[:1])]
    if not (all(c >= -1e-9 for c in crosses) or all(c <= 1e-9 for c in crosses)):
        return None
    h = config['h']; denominator = h[6]*u+h[7]*v+1
    if abs(denominator) < 1e-8:
        return None
    x,y = (h[0]*u+h[1]*v+h[2])/denominator, (h[3]*u+h[4]*v+h[5])/denominator
    if not (math.isfinite(x) and math.isfinite(y) and -5 <= x <= 5 and 0 <= y <= 8):
        return None
    return [round(x,3),round(y,3)]
