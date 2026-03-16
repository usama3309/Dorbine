import cv2
import numpy as np
import csv
import math
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_SOURCE   = 'TopDown/TopDown/clockwise_30rpm.mp4'
CSV_FILE       = 'clockwise_30rpm_analyzed_data.csv'
EVIDENCE_FOLDER = 'csv_evidence'

# Deflection pixel threshold (we set as 10 pixel threashold if the blade bend more than 10 pixel it will show red)
DEFLECTION_THRESHOLD = 10.0


# Horizontal zone gate: blade is "near horizontal" when its CSV angle
# is within this many radians of 0.
HORIZONTAL_THRESHOLD_RAD = math.radians(15)   # ±15° degrees (wake up and look at the blade when it gets within 15 degrees of horizontal)

# Quality gate: only commit a result if the best angle found is this
# close to perfect horizontal.
BEST_ANGLE_THRESHOLD_RAD = math.radians(5)    # ±5° degrees (only do the massive calculations if the blade is horizontal)

# How many frames to show the result summary before resuming playback
RESULT_DISPLAY_FRAMES = 45  # keep the warning text on the screen for 45 frames so human can read it

# Background subtractor: how many frames to warm up the model
BG_INIT_FRAMES = 60 # spend the first 60 frames of the video just leaning what the background is.



def get_blade_mask(frame, median_bg):
    # Absolute difference between current frame and empty background
    diff = cv2.absdiff(frame, median_bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Any pixel that differs by more than 30 intensity is considered moving (the blade)
    # This prevents the mask from breaking up in shaded parts of the blade!
    _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Slight blur to soften edges
    fg_mask = cv2.medianBlur(fg_mask, 5)
    
    # Slightly larger noise kernel to aggressively remove small specks (clouds/birds)
    k_noise  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
    k_fill   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    
    fg_mask  = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k_noise) # remove small noise
    fg_mask  = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_fill) # fill the gaps
    
    return fg_mask


#==========================
# this function is used to find the largest contour in the mask 
# (even after cleanup, there might be some noise left, so we find the largest contour 
# to make sure we are working with the blade)
#==========================
def find_largest_contour(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            _, (w, h), _ = cv2.minAreaRect(cnt)
            if w == 0 or h == 0: continue
            if max(w, h) / min(w, h) > 2.0 and area > best_area:
                best_area, best = area, cnt
    return best


#=================
# here is the heavy maths where we are doing all the operations, it take solid 
# white blade shape find its exact center line and then measure the deflection
#================

def get_blade_centerline_analysis(contour, mask):
    """
    Core analysis function — extracts the actual blade centerline curve and
    measures how much the TIP bends away from the ROOT's straight alignment.

    Steps:
      1. Find the blade axis via fitLine; project every contour point onto it
         to get the exact physical root (P1, min-projection) and tip (P3, max).
      2. Rotate the mask so the blade lies horizontally, scan each column for
         the vertical centre → raw centerline in rotated space.  Points are
         already ordered root→tip because we scan columns left-to-right.
      3. Smooth the raw centerline, back-transform to original image space,
         then re-sort points by their projection onto the blade axis so the
         blue polyline always flows root→tip without zigzagging.
      4. Stitch the exact P1 and P3 contour extremes to the two ends.
      5. Find P2 = the sample point closest to the physical arc-length midpoint
         of the blade (halfway between P1 and P3 along the blade axis).
      6. Fit a straight reference line ONLY on the root half (P1→P2 points),
         then measure the perpendicular deviation of the tip half (P2→P3) from
         that line → maximum deviation = deflection in pixels.
    """
    # ---- 1. Exact Physical Tip and Root via fitLine Projection ----
    lv = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = float(lv[0]), float(lv[1]), float(lv[2]), float(lv[3])
    mag = math.sqrt(vx*vx + vy*vy)
    if mag < 1e-6:
        return None
    vx, vy = vx / mag, vy / mag   # unit direction along blade axis

    pts = contour.reshape(-1, 2).astype(np.float32)
    # Scalar projection of every contour point onto the blade direction
    t_vals = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
    idx_min = int(np.argmin(t_vals))
    idx_max = int(np.argmax(t_vals))
    t_min = float(t_vals[idx_min])
    t_max = float(t_vals[idx_max])

    # P1 = root (smallest projection), P3 = tip (largest projection)
    P1_exact = (int(pts[idx_min, 0]), int(pts[idx_min, 1]))
    P3_exact = (int(pts[idx_max, 0]), int(pts[idx_max, 1]))

    # ---- 2. Centerline Extraction (Rotated Mask) ----
    angle_deg = np.degrees(np.arctan2(vy, vx))
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    h, w = mask.shape
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rot_mask = cv2.warpAffine(mask, rot_mat, (w, h))

    coords = cv2.findNonZero(rot_mask)
    if coords is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(coords)
    if rw < 5:
        return None

    step = max(1, rw // 60)
    raw_pts = []
    
    # Ignore the first 15% of the bounding box width to skip the thick, 
    # asymmetrical root/hub area which distorts the centerline trace. (we skip the firs 15% (0.15) we do this becasue root connector hub is very thick and it distorts the centerline trace
    start_x = rx + int(0.15 * rw)
    
    for sx in range(start_x, rx + rw, step):
        col = rot_mask[ry:ry + rh, sx]
        wp = np.where(col > 0)[0]
        if len(wp) > 0:
            # Find contiguous blocks of white pixels in this column
            # This prevents scattered background noise (like a chair leg)
            # from dragging the midpoint upwards.
            breaks = np.where(np.diff(wp) > 1)[0] + 1
            chunks = np.split(wp, breaks)
            
            # Find the longest solid chunk of white pixels (the actual blade body)
            longest_chunk = max(chunks, key=len)
            
            # Only count this column if the solid chunk isn't just a tiny spec
            if len(longest_chunk) > 3:
                y_mid = ry + (longest_chunk[0] + longest_chunk[-1]) / 2.0
                raw_pts.append([float(sx), y_mid])

    if len(raw_pts) < 5:
        return None
    raw_pts = np.array(raw_pts, dtype=np.float32)

    # Smooth centerline in rotated space (running-average)
    win = max(3, len(raw_pts) // 8)
    if win % 2 == 0:
        win += 1
    kernel   = np.ones(win) / win
    sm_x     = np.convolve(raw_pts[:, 0], kernel, mode='valid') # pixel dots look jagged like stairs. this does the smoothing them into fluid, mathermatically clean curve (see line by line explanation for this)
    sm_y     = np.convolve(raw_pts[:, 1], kernel, mode='valid') # same as above
    smooth_pts = np.stack([sm_x, sm_y], axis=1).astype(np.float32)
    if len(smooth_pts) < 3:
        smooth_pts = raw_pts

    # Back-transform to original image space (we un-rotate) the dots and place them back in diagonally pointing blade in the original video
    inv_rot = cv2.invertAffineTransform(rot_mat)
    orig_pts = cv2.transform(
        smooth_pts.reshape(-1, 1, 2), inv_rot
    ).reshape(-1, 2)   # shape (N, 2)

    # ---- 3. Sort orig_pts by their projection onto the blade axis ----
    # This guarantees the blue polyline flows root→tip without zigzagging
    # even if the inverse-transform reorders points slightly.
    projections = (orig_pts[:, 0] - x0) * vx + (orig_pts[:, 1] - y0) * vy
    sort_order = np.argsort(projections)
    orig_pts   = orig_pts[sort_order]

    # ---- 4. Stitch exact physical root & tip to the ends ----
    # Remove any sample points that lie outside the [P1, P3] projection range
    # (they are artifacts of the rotation/smoothing pipeline).
    projections = (orig_pts[:, 0] - x0) * vx + (orig_pts[:, 1] - y0) * vy
    mask_valid  = (projections >= t_min) & (projections <= t_max)
    orig_pts    = orig_pts[mask_valid]
    if len(orig_pts) < 2:
        return None

    orig_pts = np.vstack([
        np.array([[P1_exact[0], P1_exact[1]]], dtype=np.float32),
        orig_pts,
        np.array([[P3_exact[0], P3_exact[1]]], dtype=np.float32),
    ])

    # ---- 5. Find P2 = physical midpoint of the centerline (for display) ----
    t_mid    = (t_min + t_max) / 2.0
    all_proj = (orig_pts[:, 0] - x0) * vx + (orig_pts[:, 1] - y0) * vy
    half_idx = int(np.argmin(np.abs(all_proj - t_mid)))
    half_idx = max(1, min(half_idx, len(orig_pts) - 2))

    P1 = P1_exact
    P2 = tuple(orig_pts[half_idx].astype(int))
    P3 = P3_exact

    # ---- 6. Measure how much the BLUE LINE bends between P2 and P3 ----
    # Build the straight P2→P3 baseline and measure the perpendicular distance
    # of the tip-half centerline points from it.
    
    dx = float(P3_exact[0] - P2[0])
    dy = float(P3_exact[1] - P2[1])
    line_len = math.sqrt(dx * dx + dy * dy)
    
    if line_len < 1e-6:
        # P2 and P3 are too close, return 0 deflection
        A, B, C = 0.0, 1.0, -float(P2[1])
        deflection = 0.0
        max_dev_pt = P2
    else:
        # Line equation Ax + By + C = 0  (normal form from two points)
        A =  dy / line_len
        B = -dx / line_len
        C = -(A * P2[0] + B * P2[1])
        
        # Only check the points from P2 to P3
        tip_half_pts = orig_pts[half_idx:]
        cx_tip = tip_half_pts[:, 0]
        cy_tip = tip_half_pts[:, 1]
        
        distances  = np.abs(A * cx_tip + B * cy_tip + C)
        max_idx    = int(np.argmax(distances))
        deflection = float(distances[max_idx])
        max_dev_pt = tuple(tip_half_pts[max_idx].astype(int))

    ref_line = (A, B, C)  # kept for the cyan arrow projection

    return {
        'centerline_orig': orig_pts,   # sorted root→tip — the BLUE line
        'P1': P1,
        'P2': P2,
        'P3': P3,
        'ref_line': ref_line,
        'deflection': deflection,
        'max_dev_point': max_dev_pt,
    }







# ============================================================
# VISUALISATION  (identical to main.py)
# ============================================================

def draw_deflection_viz(frame, analysis, deflection, is_deflected, angle_deg, label):
    """
    Draw the blade bending visualisation:
      BLUE  curve  — actual blade centerline (root P1 → mid P2 → tip P3).
                     Deflection is calculated on the P2 → P3 segment.
      CYAN  arrow  — points from the maximum-bend location on the tip-half 
                     back to the P2→P3 straight baseline.
    """
    display = frame.copy()
    cl      = analysis['centerline_orig']
    P1      = analysis['P1']
    P2      = analysis['P2']
    P3      = analysis['P3']
    max_pt  = analysis['max_dev_point']
    A, B, C = analysis['ref_line']

    # ---- Helper: project a point onto the P2→P3 straight baseline ----
    def project_to_line(pt):
        """Orthogonal projection of pt onto the ref line Ax+By+C=0."""
        x, y  = float(pt[0]), float(pt[1])
        t     = A * x + B * y + C    # signed distance (A,B already unit)
        px    = x - A * t
        py    = y - B * t
        return (int(round(px)), int(round(py)))

    # ---- BLUE polyline — actual blade centerline drawn on the blade ----
    pts_int = cl.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(display, [pts_int], False, (255, 80, 0), 3)    # BLUE curve
    cv2.putText(display, "Blue: Blade shape",
                (P1[0] + 12, P1[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)

    # ---- CYAN arrow — shows the point of maximum bending ----
    max_proj = project_to_line(max_pt)
    cv2.arrowedLine(display, max_pt, max_proj, (0, 220, 220), 2, tipLength=0.25)
    cv2.putText(display, f"Max dev",
                (max_pt[0] + 8, max_pt[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)

    # ---- Dots at P1, P2, P3 ----
    cv2.circle(display, P1, 7, (0, 220, 0), -1)
    cv2.circle(display, P2, 7, (255, 255, 0), -1)
    cv2.circle(display, P3, 7, (0, 220, 220), -1)
    cv2.putText(display, "P1(root)", (P1[0] - 20, P1[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    cv2.putText(display, "P2(mid)",  (P2[0] - 20, P2[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(display, "P3(tip)",  (P3[0] - 20, P3[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

    # ---- Status overlay ----
    color  = (0, 0, 255) if is_deflected else (0, 255, 0)
    status = "DEFLECTED!" if is_deflected else "NORMAL"
    cv2.putText(display, f"Max bend: {deflection:.2f} px", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(display, f"Status: {status}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(display, f"Blade angle: {angle_deg:.2f} deg", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, label, (30, display.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return display

def save_evidence(frame, frame_num, blade_num, deflection, angle_deg):
    os.makedirs(EVIDENCE_FOLDER, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (f"{EVIDENCE_FOLDER}/blade{blade_num}_frame{frame_num}"
            f"_{ts}_{deflection:.2f}px_angle{angle_deg:.2f}.png")
    cv2.imwrite(path, frame)
    print(f"  ✓ Evidence saved: {path}")

# ============================================================
# CSV LOADING
# ============================================================

def normalise_angle(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def load_csv(filepath):
    """
    Returns a dict keyed by frame number (int) → blade angle dict.
    Only frame numbers present in the CSV are included.
    """
    frame_map = {}
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fnum = int(row['Frame'])
            frame_map[fnum] = {
                'blade1': float(row['Blade1_Angle_Rad']),
                'blade2': float(row['Blade2_Angle_Rad']),
                'blade3': float(row['Blade3_Angle_Rad']),
                'time':   float(row['Time']),
            }
    return frame_map

# ============================================================
# MAIN  — real-time playback loop
# ============================================================

def main():
    print("=" * 70)
    print("WIND TURBINE — REAL-TIME CSV + VIDEO DEFLECTION DETECTOR")
    print("=" * 70)
    print(f"Video  : {VIDEO_SOURCE}")
    print(f"CSV    : {CSV_FILE}")
    print("Controls: Q = quit | SPACE = pause/resume")
    print("=" * 70 + "\n")

    # ---- Load CSV ----
    if not os.path.isfile(CSV_FILE):
        print(f"ERROR: CSV not found: {CSV_FILE}"); return
    frame_map = load_csv(CSV_FILE)
    print(f"CSV loaded — {len(frame_map)} frame entries.\n")

    # ---- Open video ----
    if not os.path.isfile(VIDEO_SOURCE):
        print(f"ERROR: Video not found: {VIDEO_SOURCE}"); return
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_SOURCE}"); return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay_ms     = max(1, int(1000 / fps))
    print(f"Video: {total_frames} frames @ {fps:.1f} fps  (delay {delay_ms} ms/frame)\n")

    # ---- Compute Perfect Empty Background ----
    # Taking the median of 50 samples across the video perfectly erases the moving blades.
    print("Computing Median Background (takes a few seconds)...")
    frame_indices = np.linspace(0, total_frames - 1, 50, dtype=int)
    frames_for_bg = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: frames_for_bg.append(frame)
    median_bg = np.median(frames_for_bg, axis=0).astype(np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset to start
    print("Median Background ready.\n")

    # ---- State machine — one per blade ----
    BLADE_KEYS = ['blade1', 'blade2', 'blade3']
    NUM_BLADES = len(BLADE_KEYS)

    # IDLE → PEAK_SCAN (when angle enters zone) → commit best → COOLDOWN → IDLE
    IDLE, PEAK_SCAN, COOLDOWN = 0, 1, 2

    states = [{'state': IDLE, 'best_angle': 999.0, 'frame_num': None, 
               'img': None, 'mask': None, 'cooldown': 0, 
               'res_frame': None, 'res_count': 0} for _ in range(NUM_BLADES)]

    # Cooldown: skip ~half a blade-period after each trigger
    frames_per_blade_pass = max(10, int((fps * 60) / (8 * NUM_BLADES) * 0.5))

    frame_count      = 0
    deflection_count = 0
    paused           = False

    print("--- Processing... (Q = quit) ---\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video — looping.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for st in states:
                    st.update({'state': IDLE, 'best_angle': 999.0, 'frame_num': None, 
                               'img': None, 'mask': None, 'cooldown': 0,
                               'res_frame': None, 'res_count': 0})
                continue

            frame_count += 1

            # ---- Background subtractor ----
            mask = get_blade_mask(frame, median_bg)

            # ---- Look up CSV angles for this frame ----
            csv_row = frame_map.get(frame_count, None)

            # ---- Per-blade state machine ----
            for b, key in enumerate(BLADE_KEYS):
                st = states[b]

                # Cooldown countdown
                if st['state'] == COOLDOWN:
                    st['cooldown'] -= 1
                    if st['cooldown'] <= 0:
                        st['state'] = IDLE
                    continue   # skip detection during cooldown

                if csv_row is None:
                    continue   # frame not in CSV, skip

                raw_angle  = csv_row[key]
                angle_norm = normalise_angle(raw_angle)
                dist       = abs(angle_norm)   # distance from 0 (horizontal)

                if st['state'] == IDLE:
                    if dist < HORIZONTAL_THRESHOLD_RAD:
                        st['state']      = PEAK_SCAN
                        st['best_angle'] = dist
                        st['frame_num']  = frame_count
                        st['img']        = frame.copy()
                        st['mask']       = mask.copy()
                        print(f"[PEAK_SCAN] Blade {b+1} | frame {frame_count} | "
                              f"angle {math.degrees(angle_norm):.2f}°")

                elif st['state'] == PEAK_SCAN:
                    if dist < HORIZONTAL_THRESHOLD_RAD:
                        # Still in zone — update best if improved
                        if dist < st['best_angle']:
                            st['best_angle'] = dist
                            st['frame_num']  = frame_count
                            st['img']        = frame.copy()
                            st['mask']       = mask.copy()
                    else:
                        # Left the zone — process the stored best frame
                        best_a_deg = math.degrees(st['best_angle'])
                        print(f"[EXIT] Blade {b+1} | best frame: "
                              f"{st['frame_num']} | Δhoriz: {best_a_deg:.2f}°")

                        if st['best_angle'] < BEST_ANGLE_THRESHOLD_RAD:
                            # ---- Run image analysis on the best stored frame ----
                            bf      = st['img']
                            bm      = st['mask']
                            contour = find_largest_contour(bm)

                            if contour is not None:
                                analysis = get_blade_centerline_analysis(contour, bm)

                                if analysis is not None:
                                    deflection = analysis['deflection']
                                    is_deflected = deflection > DEFLECTION_THRESHOLD

                                    # Blade angle from fitted line (for display)
                                    lv = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                                    img_angle_deg = abs(np.degrees(np.arctan2(lv[1], lv[0])))
                                    if img_angle_deg > 90:
                                        img_angle_deg = abs(img_angle_deg - 180)

                                    # Build annotated result frame
                                    vis = bf.copy()
                                    overlay = vis.copy(); overlay[bm > 0] = (0, 255, 0)
                                    vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)
                                    label = (f"Blade {b+1} | CSV frame {st['frame_num']} "
                                             f"| Δhoriz={best_a_deg:.2f}°")
                                    result_frame = draw_deflection_viz(
                                        vis, analysis, deflection, is_deflected,
                                        img_angle_deg, label)

                                    save_evidence(result_frame, st['frame_num'],
                                                  b + 1, deflection, img_angle_deg)

                                    st['res_frame'] = result_frame
                                    st['res_count'] = RESULT_DISPLAY_FRAMES

                                    if is_deflected:
                                        deflection_count += 1
                                        print(f"  DEFLECTED  | {deflection:.2f} px")
                                    else:
                                        print(f"  NORMAL     | {deflection:.2f} px")
                                else:
                                    print(f"  [SKIP] Centerline extraction failed.")
                            else:
                                print(f"  [SKIP] No contour in best frame.")
                        else:
                            print(f"  [SKIP] Δhoriz {best_a_deg:.2f}° > "
                                  f"quality gate {math.degrees(BEST_ANGLE_THRESHOLD_RAD):.1f}°")

                        # Enter cooldown
                        st['state']      = COOLDOWN
                        st['cooldown']   = frames_per_blade_pass
                        st['best_angle'] = 999.0
                        st['frame_num']  = None
                        st['img']        = None
                        st['mask']       = None

            # ---- Build display frame ----
            display = frame.copy()

            # Green mask overlay on live feed
            overlay = display.copy(); overlay[mask > 0] = (0, 255, 0)
            display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

            # Per-blade state labels
            for b, key in enumerate(BLADE_KEYS):
                s_name = ["IDLE", "PEAK_SCAN", "COOLDOWN"][states[b]['state']]
                if csv_row:
                    ang = math.degrees(normalise_angle(csv_row[key]))
                    txt = f"B{b+1}: {s_name}  ({ang:.1f}°)"
                else:
                    txt = f"B{b+1}: {s_name}  (no CSV)"
                cv2.putText(display, txt, (30, 50 + b * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(display, f"Frame: {frame_count}", (30, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # ---- Show result overlays (show each blade's result for N frames) ----
            for b in range(NUM_BLADES):
                st = states[b]
                if st['res_count'] > 0 and st['res_frame'] is not None:
                    win = f"Blade {b+1} Deflection Result"
                    cv2.imshow(win, st['res_frame'])
                    st['res_count'] -= 1
                    if st['res_count'] == 0:
                        cv2.destroyWindow(win)

            cv2.imshow("Live Feed", display)
            #cv2.imshow("Mask", mask)

        # ---- Keyboard ----
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            print("\nQ pressed — quitting.")
            break
        elif key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print(f"  Frames processed : {frame_count}")
    print(f"  Deflections found: {deflection_count}")
    print(f"  Evidence saved in: {EVIDENCE_FOLDER}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
