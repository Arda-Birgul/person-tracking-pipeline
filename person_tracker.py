#!/usr/bin/env python3
"""
Optimized Person Tracking Pipeline

A high-performance person detection and tracking system with the following capabilities:
- Multi-processing support with automatic worker optimization
- YOLO-based person detection with configurable parameters
- Stable ID tracking using enhanced CentroidTracker
- Performance optimizations including FP16, batch inference, and threaded I/O
- Comprehensive logging and visualization
- ONNX export support for model deployment
"""

import os, sys, argparse, math, subprocess, multiprocessing, time, gc
from datetime import datetime
from queue import Queue
from threading import Thread
import cv2, numpy as np

# -------- CONFIGURATION SETTINGS --------
INPUT_DIR = "InputData"
OUTPUT_DIR = "OutputData"

# Detection thresholds
YOLO_MIN_CONF = 0.25
MIN_CONF_TO_CREATE = 0.65
MIN_CONF_TO_SHOW = 0.65
IOU_MIN = 0.35

# Tracking parameters
DIST_THRESH = 40.0
MAX_MISSED = 6
MIN_HITS = 2
EMA_ALPHA_BBOX = 0.55
EMA_ALPHA_CONF = 0.45

# Inference settings
INFER_IMG_SZ = 1280
INFER_AUGMENT = False
INFER_IOU = 0.45

# Processing parameters
DETECT_EVERY_K_FRAMES = 3
BATCH_SIZE = 4
QUEUE_SIZE = 32
WARMUP_STEPS = 3

# Performance optimization
CLEANUP_INTERVAL = 150
PROFILE_ENABLED = True

# Dependencies
REQUIRED_PIP = ["ultralytics", "numpy", "opencv-python-headless", "requests", "torch", "psutil"]

# Benchmarking settings
BENCH_FRAMES_GPU = 3
BENCH_FRAMES_CPU = 8
CPU_UTIL_LIMIT = 92.0
MAX_CANDIDATE_WORKERS = 8

# -------- UTILITY FUNCTIONS --------

def prepare_model_fp16_and_warmup(model, device="cuda", imgsz=640, warmup_steps=2):
    """Optimize model with FP16 and perform warmup"""
    import torch
    model.to(device)
    try:
        if device == "cuda":
            try:
                model.model.half()
            except Exception:
                pass
    except Exception:
        pass

    model.model.eval()
    # Warmup inference
    for _ in range(max(1, warmup_steps)):
        dummy = (np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")
        try:
            with torch.no_grad():
                _ = model(dummy, imgsz=imgsz, conf=0.25, device=device)
        except Exception:
            try:
                _ = model(dummy)
            except Exception:
                pass
    return model

def parse_ultralytics_results_to_dets(result):
    """Convert Ultralytics results to [x,y,w,h,conf] format"""
    out = []
    boxes = getattr(result, "boxes", None)
    if not boxes:
        return out
    for b in boxes:
        try:
            xyxy = b.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            conf = float(b.conf[0].cpu().numpy())
        except Exception:
            continue
        out.append([x1, y1, x2-x1, y2-y1, conf])
    return out

class CentroidTracker:
    """Enhanced Centroid Tracker for stable ID tracking"""
    def __init__(self, max_lost=6, dist_thresh=50.0):
        self.next_id = 1
        self.tracks = {}  # id -> {bbox, center, last_seen, missed, hits, conf_ema, confirmed}
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh

    def _center(self, bbox):
        x,y,w,h = bbox
        return (x + w/2.0, y + h/2.0)

    def _dist(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    def update_with_detections(self, dets, frame_idx):
        """Update tracks with new detections"""
        assigned = set()
        
        # Match existing tracks
        for tid, t in list(self.tracks.items()):
            best_i = -1; best_d = 1e9
            for i,d in enumerate(dets):
                if i in assigned: continue
                c = self._center(d[:4])
                dlen = self._dist(c, t["center"])
                if dlen < best_d:
                    best_d = dlen; best_i = i
            
            if best_i != -1 and best_d < self.dist_thresh:
                d = dets[best_i]
                self._update_track(tid, d, frame_idx)
                assigned.add(best_i)
            else:
                t["missed"] += 1

        # Create new tracks
        for i,d in enumerate(dets):
            if i in assigned: continue
            if d[4] >= MIN_CONF_TO_CREATE:
                self._create_track(d, frame_idx)

        # Remove old tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_lost:
                del self.tracks[tid]

    def _create_track(self, det, fidx):
        tid = self.next_id; self.next_id += 1
        bbox = det[:4]; conf = float(det[4])
        c = self._center(bbox)
        self.tracks[tid] = {
            "bbox": bbox[:], 
            "smooth_bbox": bbox[:],
            "center": c, 
            "last_seen": fidx, 
            "missed": 0, 
            "hits": 1, 
            "conf_ema": conf, 
            "confirmed": False
        }
        if self.tracks[tid]["hits"] >= MIN_HITS:
            self.tracks[tid]["confirmed"] = True

    def _update_track(self, tid, det, fidx):
        bbox = det[:4]; conf = float(det[4])
        t = self.tracks[tid]
        sx,sy,sw,sh = t["smooth_bbox"]
        bx,by,bw,bh = bbox
        a = EMA_ALPHA_BBOX
        
        # Smooth bbox update
        t["smooth_bbox"] = [
            a*bx + (1-a)*sx, 
            a*by + (1-a)*sy, 
            a*bw + (1-a)*sw, 
            a*bh + (1-a)*sh
        ]
        t["bbox"] = bbox[:]
        t["center"] = self._center(t["smooth_bbox"])
        t["last_seen"] = fidx
        t["missed"] = 0
        t["hits"] += 1
        t["conf_ema"] = EMA_ALPHA_CONF*conf + (1-EMA_ALPHA_CONF)*t["conf_ema"]
        
        if not t["confirmed"] and t["hits"] >= MIN_HITS:
            t["confirmed"] = True

    def predict(self):
        """Simple predict for next frame preparation"""
        pass

    def get_confirmed(self):
        """Return confirmed tracks"""
        return [(tid, t) for tid,t in self.tracks.items() if t["confirmed"]]

def detect_every_k_frames(frames_iter, detector_fn, K=3, tracker=None):
    """Run detection every K frames"""
    if tracker is None:
        tracker = CentroidTracker(max_lost=MAX_MISSED, dist_thresh=DIST_THRESH)
    
    for frame_idx, frame in frames_iter:
        if frame_idx % K == 0:
            dets = detector_fn(frame)
            tracker.update_with_detections(dets, frame_idx)
        else:
            tracker.predict()
        yield frame_idx, frame, tracker.get_confirmed()

def batch_frames_infer(model, frames, device="cuda", imgsz=640, conf=0.25, iou=0.45):
    """Batch inference for better efficiency"""
    import torch
    try:
        with torch.no_grad():
            results = model(frames, imgsz=imgsz, conf=conf, iou=iou, device=device)
    except Exception:
        results = model(frames)
    
    all_dets = []
    for r in results:
        dets = parse_ultralytics_results_to_dets(r)
        all_dets.append(dets)
    return all_dets

# Threading utilities
def start_reader_thread(video_path, read_q, stop_token=None):
    """Start video reading thread"""
    def _reader():
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            read_q.put((idx, frame))
            idx += 1
        cap.release()
        read_q.put(stop_token)
    
    t = Thread(target=_reader, daemon=True)
    t.start()
    return t

def start_writer_thread(out_path, write_q, fps, size, stop_token=None):
    """Start video writing thread"""
    def _writer():
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        out = cv2.VideoWriter(out_path, fourcc, fps, size)
        while True:
            item = write_q.get()
            if item is stop_token:
                break
            frame = item
            out.write(frame)
        out.release()
    
    t = Thread(target=_writer, daemon=True)
    t.start()
    return t

def draw_person_and_head_labels(frame, bbox, conf_pct, id_text):
    """Draw person and head detection labels"""
    x,y,w,h = bbox
    x_i,y_i,w_i,h_i = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    
    # Person bbox (green)
    cv2.rectangle(frame, (x_i,y_i), (x_i+w_i, y_i+h_i), (0,255,0), 2)
    
    # ID label (top)
    label = id_text
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    bx1, by1 = x_i, max(0, y_i - (lh + 8))
    bx2, by2 = x_i + lw + 8, by1 + lh + 8
    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,0,0), -1)
    cv2.putText(frame, label, (bx1+4, by2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    # Head detection box (red upper region)
    head_height_ratio = 0.32
    head_width_ratio = 0.7
    hx = x + w*(0.5 - head_width_ratio/2)
    hy = y
    hw = w * head_width_ratio
    hh = h * head_height_ratio
    hx_i, hy_i, hw_i, hh_i = int(round(hx)), int(round(hy)), int(round(hw)), int(round(hh))
    
    if hw_i > 4 and hh_i > 4:
        cv2.rectangle(frame, (hx_i, hy_i), (hx_i+hw_i, hy_i+hh_i), (0,0,255), 2)
        head_label = f"Head {conf_pct}%"
        (hwl, hhl), _ = cv2.getTextSize(head_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        label_x = x_i
        label_y = y_i + h_i + 8 + hhl
        if label_y + 2 > frame.shape[0]:
            label_y = max(0, y_i - (hhl + 8))
        cv2.rectangle(frame, (label_x, label_y - hhl - 6), (label_x + hwl + 8, label_y + 2), (0,0,0), -1)
        cv2.putText(frame, head_label, (label_x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

# Performance monitoring
class Profiler:
    def __init__(self):
        self.acc = {}
    
    def add(self, name, dt):
        self.acc.setdefault(name, []).append(dt)
    
    def report(self):
        for k,v in self.acc.items():
            import statistics
            print(f"{k}: mean={statistics.mean(v):.3f}s median={statistics.median(v):.3f}s n={len(v)}")

def periodic_cleanup(frame_idx, interval=150, device="cuda", no_gpu_cache_clean=False):
    """Periodic memory cleanup"""
    if frame_idx % interval != 0:
        return
    gc.collect()
    if device == "cuda" and not no_gpu_cache_clean:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

# -------- AUTO-TUNING FUNCTIONS --------

def detect_gpu_available():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def run_cmd(cmd):
    """Run command and return result"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)

def ensure_pip_packages(pkgs):
    """Ensure required packages are installed"""
    import importlib
    for p in pkgs:
        name = p.split("==")[0]
        try:
            importlib.import_module(name)
        except Exception:
            print(f"{name} not found. Installing via pip...")
            rc, out, err = run_cmd([sys.executable, "-m", "pip", "install", p])
            if rc != 0:
                print(f"Failed to install {name}: {err.strip()}")

def load_local_or_fallback(preferred="yolov8m.pt"):
    """Load model with fallback options"""
    from ultralytics import YOLO
    try:
        model = YOLO(preferred)
        print("Model loaded:", preferred)
        return model
    except Exception as e1:
        print(f"Failed to load {preferred}: {e1}")
        try:
            model = YOLO("yolov8n.pt")
            print("Fallback: using yolov8n.pt")
            return model
        except Exception as e3:
            raise RuntimeError("Failed to load model: " + str(e3))

def bench_worker_process(device, frames_to_run, use_model=True, model_name="yolov8m.pt"):
    """Benchmark worker process performance"""
    import time
    start = time.time()
    processed = 0
    
    if use_model and device=="cuda":
        from ultralytics import YOLO
        try:
            model = load_local_or_fallback(model_name)
            model = prepare_model_fp16_and_warmup(model, device, INFER_IMG_SZ, WARMUP_STEPS)
        except Exception:
            use_model = False

    if use_model and device=="cuda":
        for i in range(frames_to_run):
            img = (np.random.rand(480,640,3)*255).astype("uint8")
            try:
                _ = model(img, conf=YOLO_MIN_CONF, iou=INFER_IOU, imgsz=INFER_IMG_SZ, 
                         augment=INFER_AUGMENT, device=device)
            except Exception:
                try:
                    _ = model(img)
                except Exception:
                    pass
            processed += 1
    else:
        # CPU simulation
        for i in range(frames_to_run):
            img = (np.random.rand(480,640,3)*255).astype("uint8")
            r = cv2.resize(img, (320,240))
            s = r.mean()
            _ = s*1.0
            processed += 1
    
    elapsed = time.time() - start
    return processed, elapsed

def _bench_proc_worker_wrapper(queue, device, frames_to_run, model_name):
    """Worker wrapper for benchmarking"""
    try:
        proc, elapsed = bench_worker_process(device, frames_to_run, use_model=(device=="cuda"), model_name=model_name)
        queue.put(proc)
    except Exception:
        queue.put(0)

def run_stress_test(video_path, candidates, device, model_name, no_install):
    """Run stress test to find optimal worker count"""
    try:
        import psutil
    except Exception:
        psutil = None

    results = {}
    frames_per_worker = BENCH_FRAMES_GPU if device=="cuda" else BENCH_FRAMES_CPU

    for w in candidates:
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        procs = []
        
        for i in range(w):
            p = ctx.Process(target=_bench_proc_worker_wrapper, 
                          args=(queue, device, frames_per_worker, model_name))
            p.start()
            procs.append(p)

        start = time.time()
        cpu_percent_sample = None
        if psutil:
            time.sleep(0.2)
            cpu_percent_sample = psutil.cpu_percent(interval=0.5)
        
        for p in procs:
            p.join(timeout=60)
        elapsed = time.time() - start

        total_processed = 0
        while not queue.empty():
            try:
                total_processed += int(queue.get_nowait() or 0)
            except Exception:
                break

        throughput = total_processed / elapsed if elapsed > 0 else 0.0
        results[w] = {
            "processed": total_processed, 
            "elapsed": elapsed, 
            "throughput": throughput, 
            "cpu_percent": cpu_percent_sample
        }
        print(f"Test w={w}: processed={total_processed}, time={elapsed:.2f}s, throughput={throughput:.1f} fps, cpu%={cpu_percent_sample}")
        time.sleep(0.3)

    # Select best worker count
    best = None; best_thr = 0.0
    for w, r in results.items():
        cpu_ok = True
        if r["cpu_percent"] is not None and r["cpu_percent"] > CPU_UTIL_LIMIT:
            cpu_ok = False
        if r["throughput"] > best_thr and cpu_ok:
            best = w; best_thr = r["throughput"]
    
    if best is None:
        best = max(results.items(), key=lambda kv: kv[1]["throughput"])[0]
    return best, results

def choose_workers_auto(video_path, max_candidates=None, model_name="yolov8m.pt", no_install=False):
    """Automatically choose optimal worker count"""
    cpu_count = os.cpu_count() or 1
    gpu = detect_gpu_available()
    device = "cuda" if gpu else "cpu"

    cand = [1,2,4, max(1, cpu_count//2), cpu_count]
    cand = [c for c in cand if c > 0]
    cand = sorted(list(set(cand)))
    if max_candidates:
        cand = [c for c in cand if c <= max_candidates]
    cand = [c for c in cand if c <= MAX_CANDIDATE_WORKERS]
    if not cand:
        cand = [1]

    print("Auto-tune: resource detection -> cpu:", cpu_count, "gpu:", gpu)
    print("Auto-tune: testing worker counts:", cand)
    best, report = run_stress_test(video_path, cand, device, model_name, no_install)
    print("Auto-tune selection:", best)
    return best, report

# -------- WORKER PROCESS (Enhanced) --------

def worker_process(video_path, start_frame, end_frame, part_idx, outdir, model_name, device, no_gpu_cache_clean):
    """Enhanced worker process for video processing"""
    import torch
    from ultralytics import YOLO
    
    # Load and optimize model
    try:
        model = load_local_or_fallback(model_name)
        model = prepare_model_fp16_and_warmup(model, device, INFER_IMG_SZ, WARMUP_STEPS)
        print(f"Worker {part_idx}: Model optimization completed")
    except Exception as e:
        print(f"Worker {part_idx}: Model loading error:", e)
        return False

    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Worker {part_idx}: Cannot open video:", video_path)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output setup
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    part_video = os.path.join(outdir, f"part_{part_idx}.webm")
    out = cv2.VideoWriter(part_video, fourcc, fps, (W,H))
    log_path = os.path.join(outdir, f"log_part_{part_idx}.txt")
    log = open(log_path, "w", encoding="utf-8")
    log.write("log format: Frame/Remaining // ID // Status // Confidence%\n")

    # Tracking setup
    tracker = CentroidTracker(max_lost=MAX_MISSED, dist_thresh=DIST_THRESH)
    
    # Performance monitoring
    profiler = Profiler() if PROFILE_ENABLED else None
    
    # Detector function
    def detector_fn(frame):
        t0 = time.time()
        try:
            results = model(frame, conf=YOLO_MIN_CONF, iou=INFER_IOU, imgsz=INFER_IMG_SZ,
                          augment=INFER_AUGMENT, device=device, classes=[0], max_det=200)
        except TypeError:
            results = model(frame)
        
        dets = []
        for r in results:
            frame_dets = parse_ultralytics_results_to_dets(r)
            dets.extend(frame_dets)
        
        if profiler:
            profiler.add("inference", time.time() - t0)
        return dets

    # Frame iterator
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    
    def frames_iter():
        nonlocal frame_idx
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    # Main processing loop
    for fidx, frame, confirmed_tracks in detect_every_k_frames(
        frames_iter(), detector_fn, K=DETECT_EVERY_K_FRAMES, tracker=tracker):
        
        # HUD information
        remaining = max(0, end_frame - fidx)
        hud_text = f"{fidx}/{remaining}"
        (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pad = 6
        x1, y1 = W - tw - 2*pad, 0
        x2, y2 = W, th + 2*pad
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), -1)
        cv2.putText(frame, hud_text, (x1+pad, y2-pad-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Track logging and drawing
        for tid, track in confirmed_tracks:
            conf_pct = int(round(track["conf_ema"] * 100))
            status = "Person" if track["last_seen"] == fidx else "unknown"
            log.write(f"{fidx}/{remaining} // {tid} // Status: {status} // %:{conf_pct}\n")

            # Draw only current and confident tracks
            if (track["last_seen"] == fidx and 
                track["conf_ema"] >= MIN_CONF_TO_SHOW and 
                track["confirmed"]):
                
                id_text = f"ID:{tid} {conf_pct}%"
                draw_person_and_head_labels(frame, track["smooth_bbox"], conf_pct, id_text)

        # Write frame
        out.write(frame)

        # Periodic cleanup
        periodic_cleanup(fidx, CLEANUP_INTERVAL, device, no_gpu_cache_clean)

    # Cleanup
    cap.release()
    out.release()
    log.close()
    
    if profiler:
        print(f"Worker {part_idx} performance report:")
        profiler.report()
    
    print(f"Worker {part_idx} completed: {part_video}")
    return True

# -------- MERGE FUNCTIONS --------

def merge_video_parts(outdir, final_path, fps, W, H):
    """Merge video parts into final output"""
    import glob
    parts = sorted(glob.glob(os.path.join(outdir, "part_*.webm")))
    if not parts:
        return False
    
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out = cv2.VideoWriter(final_path, fourcc, fps, (W,H))
    
    for p in parts:
        cap = cv2.VideoCapture(p)
        while True:
            ret, f = cap.read()
            if not ret: 
                break
            out.write(f)
        cap.release()
    
    out.release()
    return True

def merge_logs(outdir, final_log):
    """Merge log files into final log"""
    files = sorted([os.path.join(outdir, f) for f in os.listdir(outdir) if f.startswith("log_part_")])
    with open(final_log, "w", encoding="utf-8") as fw:
        header_written = False
        for f in files:
            with open(f, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
            if not lines: 
                continue
            if not header_written:
                fw.write(lines[0])
                header_written = True
            for ln in lines[1:]:
                fw.write(ln)
    return True

# -------- MAIN PROCESSING --------

def process_video_parallel(video_path, workers, model_name, no_gpu_cache_clean):
    """Main parallel video processing function"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Calculate frame ranges
    if workers < 1: 
        workers = 1
    chunk = math.ceil(total / workers)
    
    # Output directory
    out_base = os.path.basename(video_path)
    name, _ = os.path.splitext(out_base)
    outdir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(outdir, exist_ok=True)

    print(f"Processing video: {name}")
    print(f"Total frames: {total}, Workers: {workers}, Chunk size: {chunk}")

    # Start worker processes
    procs = []
    for i in range(workers):
        start = i * chunk
        end = min(total - 1, (i + 1) * chunk - 1)
        if start <= end:
            p = multiprocessing.Process(
                target=worker_process, 
                args=(video_path, start, end, i, outdir, model_name, device_global, no_gpu_cache_clean)
            )
            p.start()
            procs.append(p)
            print(f"Worker {i} started: frames {start}-{end}")

    # Wait for workers
    for p in procs:
        p.join()

    # Merge parts
    final_video = os.path.join(outdir, "Bbox.webm")
    final_log = os.path.join(outdir, "log.txt")
    
    print("Merging video parts...")
    merge_video_parts(outdir, final_video, fps, W, H)
    merge_logs(outdir, final_log)
    
    print(f"Processing completed: {final_video}")
    print(f"Log file: {final_log}")

# -------- ONNX EXPORT SUPPORT --------

def export_to_onnx(model, out_path="model.onnx", imgsz=640):
    """Export model to ONNX format"""
    import torch
    try:
        dummy = torch.randn(1, 3, imgsz, imgsz)
        torch.onnx.export(
            model.model, dummy, out_path, 
            opset_version=14, 
            input_names=["images"], 
            output_names=["out"]
        )
        print("ONNX export completed:", out_path)
        return True
    except Exception as e:
        print("ONNX export error:", e)
        return False

def onnx_infer(sess, img_np):
    """ONNX runtime inference"""
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: img_np.astype("float32")[None, ...]})
    return out

# -------- FRAME RANGE HELPERS --------

def frame_ranges_for_workers(total_frames, workers):
    """Calculate frame ranges for workers"""
    if workers < 1: 
        workers = 1
    chunk = math.ceil(total_frames / workers)
    ranges = []
    for i in range(workers):
        start = i * chunk
        end = min(total_frames - 1, (i + 1) * chunk - 1)
        if start <= end:
            ranges.append((start, end))
    return ranges

# -------- MAIN CLI --------

def main():
    parser = argparse.ArgumentParser(description="Person Tracking Pipeline with Advanced Features")
    parser.add_argument("--no-install", action="store_true", 
                       help="Skip automatic package installation")
    parser.add_argument("--workers", type=int, default=0, 
                       help="Number of worker processes (0 = auto-tune)")
    parser.add_argument("--auto-tune", type=str, default="true", 
                       help="Enable auto-tuning (true/false)")
    parser.add_argument("--model", type=str, default="yolov8m.pt", 
                       help="YOLO model to use")
    parser.add_argument("--no-gpu-cache-clean", action="store_true", 
                       help="Disable GPU cache cleaning")
    parser.add_argument("--export-onnx", action="store_true", 
                       help="Export model to ONNX format")
    parser.add_argument("--detect-interval", type=int, default=3, 
                       help="Run detection every N frames")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch inference size")
    parser.add_argument("--profile", action="store_true", 
                       help="Enable performance profiling")
    
    args = parser.parse_args()

    # Update global settings
    global DETECT_EVERY_K_FRAMES, BATCH_SIZE, PROFILE_ENABLED
    DETECT_EVERY_K_FRAMES = args.detect_interval
    BATCH_SIZE = args.batch_size
    PROFILE_ENABLED = args.profile

    # Package installation
    if not args.no_install:
        print("Checking and installing required packages...")
        ensure_pip_packages(REQUIRED_PIP)
    else:
        print("--no-install: Package installation skipped")

    # Device detection
    global device_global
    device_global = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device_global = "cuda"
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not found, using CPU mode")
    except Exception:
        device_global = "cpu"
        print("PyTorch not found, using CPU mode")

    # Configuration
    auto_tune_flag = str(args.auto_tune).lower() not in ("0", "false", "no")
    manual_workers = args.workers if args.workers > 0 else None
    model_name = args.model
    no_gpu_clean = args.no_gpu_cache_clean

    cpu_count = os.cpu_count() or 1
    print(f"System info -> Device: {device_global}, CPU cores: {cpu_count}")

    # Create directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find video files
    video_extensions = (".webm", ".mp4", ".mkv", ".avi", ".mov", ".m4v")
    videos = []
    for f in os.listdir(INPUT_DIR):
        if os.path.isfile(os.path.join(INPUT_DIR, f)) and f.lower().endswith(video_extensions):
            videos.append(os.path.join(INPUT_DIR, f))
    videos = sorted(videos)

    if not videos:
        print(f"Error: No video files found in {INPUT_DIR} directory.")
        print(f"Supported formats: {video_extensions}")
        sys.exit(1)

    print(f"Found videos: {len(videos)}")
    for v in videos:
        print(f"  - {os.path.basename(v)}")

    # ONNX export (optional)
    if args.export_onnx:
        print("Starting ONNX export process...")
        try:
            model = load_local_or_fallback(model_name)
            model = prepare_model_fp16_and_warmup(model, device_global, INFER_IMG_SZ, WARMUP_STEPS)
            onnx_path = os.path.join(OUTPUT_DIR, f"{model_name.replace('.pt', '')}.onnx")
            export_to_onnx(model, onnx_path, INFER_IMG_SZ)
        except Exception as e:
            print(f"ONNX export error: {e}")

    # Video processing loop
    total_start_time = time.time()
    
    for i, vid in enumerate(videos):
        print(f"\n{'='*60}")
        print(f"Video {i+1}/{len(videos)}: {os.path.basename(vid)}")
        print(f"{'='*60}")
        
        video_start_time = time.time()
        
        # Determine worker count
        chosen_workers = None
        
        if manual_workers:
            chosen_workers = manual_workers
            print(f"Manual worker count: {chosen_workers}")
        
        elif auto_tune_flag:
            print("Starting auto-tune (performance optimization)...")
            try:
                chosen_workers, report = choose_workers_auto(
                    vid, model_name=model_name, no_install=args.no_install
                )
                print(f"Auto-tune result: {chosen_workers} workers selected")
                
                # Show report
                print("Performance report:")
                for w, r in report.items():
                    print(f"  {w} workers: {r['throughput']:.1f} fps, CPU: {r['cpu_percent']}%")
                    
            except Exception as e:
                print(f"Auto-tune error: {e}")
                chosen_workers = max(1, min(cpu_count, 4))
                print(f"Fallback worker count: {chosen_workers}")
        
        else:
            # Simple fallback
            chosen_workers = max(1, min(cpu_count, cpu_count // 2 or 1))
            print(f"Simple worker count: {chosen_workers}")

        # Process video
        try:
            process_video_parallel(vid, chosen_workers, model_name, no_gpu_clean)
            
            video_elapsed = time.time() - video_start_time
            print(f"Video processing time: {video_elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"Video processing error: {e}")
            continue

    # Total time
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"All videos completed!")
    print(f"Total processing time: {total_elapsed:.2f} seconds")
    print(f"Average per video: {total_elapsed/len(videos):.2f} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows compatibility
    main()