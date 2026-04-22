# Video processing loop for GUI (to be inserted into gui.py)

def _video_loop(self) -> None:
    """Process video file frame-by-frame."""
    if not self._video_file:
        print("[VIDEO] No video file loaded!")
        self._running = False
        return
    
    cap = cv2.VideoCapture(self._video_file)
    if not cap.isOpened():
        print(f"[VIDEO] Cannot open: {self._video_file}")
        self._running = False
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / self._cfg.capture_fps if fps > 0 else 0.2
    
    print(f"[VIDEO] Processing {total_frames} frames @ {fps:.1f} fps (capture rate: {self._cfg.capture_fps} fps)")
    
    # Initialize stitcher
    if self._stitcher is None:
        cfg = self._cfg
        self._stitcher = PanoramaStitcher(cfg, on_update=self._on_stitch_update)
    
    frame_idx = 0
    processed = 0
    
    try:
        while self._running and cap.isOpened():
            ret, frame_rgb = cap.read()
            if not ret:
                print("[VIDEO] End of video")
                break
            
            frame_idx += 1
            
            # Respect capture_fps
            now = time.time()
            if now - self._last_cap_ts < frame_delay:
                continue
            self._last_cap_ts = now
            
            # Resize if needed
            h, w = frame_rgb.shape[:2]
            if w != self._cfg.width or h != self._cfg.height:
                frame_rgb = cv2.resize(frame_rgb, (self._cfg.width, self._cfg.height))
            
            # Generate thermal from RGB (or use actual thermal if available)
            heat_src = frame_rgb
            use_thermal_as_src = self._source_var.get() == "thermal"
            
            if use_thermal_as_src and self._cap_thermal:
                ret_t, frame_thermal = self._cap_thermal.read()
                if ret_t:
                    heat_src = frame_thermal
            
            thermal_gray = cv2.cvtColor(heat_src, cv2.COLOR_BGR2GRAY)
            
            # Feed to stitcher
            self._stitcher.feed_frame(frame_rgb.copy(), thermal_gray, now)
            processed += 1
            
            # Update display
            with self._frame_lock:
                self._latest_rgb = frame_rgb.copy()
            
            # Progress update
            if frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                self._root.after(0, lambda p=progress, pr=processed: 
                    self._status_var.set(f"Processing video: {p:.1f}% ({pr} frames stitched)"))
        
        print(f"[VIDEO] Processing complete: {processed} frames stitched")
        
    except Exception as e:
        print(f"[VIDEO] Error: {e}")
    finally:
        cap.release()
        self._running = False
