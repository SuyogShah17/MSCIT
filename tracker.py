import math
class EuclideanDistTracker:
    def __init__(s): s.center_points, s.id_count = {}, 0
    def update(s, rects):
        ids, new = [], {}
        for x,y,w,h in rects:
            cx, cy = x+w//2, y+h//2; found = False
            for id, pt in s.center_points.items():
                if math.hypot(cx-pt[0], cy-pt[1]) < 25:
                    new[id] = (cx, cy); ids.append([x,y,w,h,id]); found = True; break
            if not found:
                new[s.id_count] = (cx, cy); ids.append([x,y,w,h,s.id_count]); s.id_count += 1
        s.center_points = new.copy(); return ids
