# Panoramic Heat Mapping - Position Persistence & Relocalization

**Status:** CRITICAL ISSUE  
**Priority:** P0  
**Created:** 2026-04-15  
**Author:** Deep Dive Analysis via Müctehid

---

## 🔴 Problem Statement

### Current Behavior (BROKEN)
When the camera moves away from a previously scanned area and then returns:
1. ✅ Camera captures thermal data at position A (e.g., monitor back)
2. ✅ Thermal data appears on heat map
3. ❌ Camera moves to position B (monitor front)
4. ❌ Camera returns to position A
5. **❌ CRITICAL:** Previously captured thermal data at position A **disappears** from heat map

### Expected Behavior (TARGET)
1. ✅ Camera captures thermal data at position A
2. ✅ Thermal data persists in **global coordinate space**
3. ✅ Camera moves to position B
4. ✅ Camera returns to position A
5. **✅ GOAL:** System **recognizes** position A, displays last known thermal data
6. **✅ GOAL:** New thermal readings update/blend with previous data at same spatial location

---

## 🔬 Root Cause Analysis

### Current Architecture (Incremental Accumulation)

```python
# panorama_stitcher.py:486-537
self._H_acc = np.eye(3, dtype=np.float64)  # Identity at start

# For each new frame:
H, inliers = self._aligner.compute_homography(self._prev.rgb, fp.rgb)
# H maps: previous_frame → current_frame

# Accumulate transformation
self._H_acc[0, 2] += tx  # Incremental X offset
self._H_acc[1, 2] += ty  # Incremental Y offset

# Warp thermal onto canvas
H_canvas = self._H_offset @ self._H_acc
warped = cv2.warpPerspective(thermal, H_canvas, (cw, ch))
```

### Why It Fails

| Component | Current Behavior | Problem |
|-----------|------------------|---------|
| **Coordinate System** | Relative (frame-to-frame) | No global reference |
| **H_acc** | Incremental accumulation | Drift accumulates, no loop closure |
| **Position Recognition** | Only matches `_prev` frame | Can't recognize previously visited locations |
| **Canvas Mapping** | One-way: frame → canvas | No inverse: canvas → world |
| **Thermal Persistence** | Pixels written once | No spatial memory of "what was at X,Y" |

### Concrete Example

```
Frame 1 (monitor back):  H_acc = [1, 0, 0]     ← Initial position
                                  [0, 1, 0]
                                  
Frame 2 (move right):    H_acc = [1, 0, +50]   ← Accumulated +50px
                                  [0, 1, 0]
                                  
Frame 3 (move left):     H_acc = [1, 0, +50-45] ← Should be ~0, but...
                                  [0, 1, +5]      ← Drift! Not exactly Frame 1
                                  
❌ System doesn't know Frame 3 ≈ Frame 1 (same physical location)
❌ Treats it as new position → old thermal data not retrieved
```

---

## 🎯 Solution Architecture

### Approach: **Feature-Based Relocalization + Keyframe Database**

Inspired by Visual SLAM (ORB-SLAM, RTAB-Map) but simplified for 2D thermal panorama.

### Core Components

#### 1. **Keyframe Database**
```python
@dataclass
class Keyframe:
    id: int
    rgb: np.ndarray              # RGB image for feature matching
    thermal: np.ndarray          # Thermal data
    H_to_global: np.ndarray      # Transform to global coordinate system
    features: tuple              # (keypoints, descriptors) - AKAZE
    timestamp: float
    canvas_bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) on canvas
```

#### 2. **Global Coordinate System**
- **Origin:** First frame defines (0, 0) in global space
- **Canvas:** Persistent, never reset
- **All frames:** Mapped to global coords via `H_to_global`

#### 3. **Relocalization Pipeline**

```python
def process_frame(self, fp: FramePair) -> None:
    # Step 1: Try match with previous frame (normal tracking)
    H_local, inliers = self._aligner.compute_homography(self._prev.rgb, fp.rgb)
    
    if H_local is not None and inliers > MIN_INLIERS:
        # Good tracking - update H_acc incrementally
        self._H_acc = self._H_acc @ H_local
        fp.H_to_global = self._H_acc.copy()
    else:
        # Step 2: Tracking lost - try relocalize against keyframes
        best_match = self._relocalize(fp.rgb)
        
        if best_match is not None:
            # Found previous location!
            kf = self._keyframes[best_match.kf_id]
            H_curr_to_kf, inliers = self._aligner.compute_homography(
                kf.rgb, fp.rgb
            )
            # Compute global position from keyframe
            fp.H_to_global = kf.H_to_global @ H_curr_to_kf
            self._H_acc = fp.H_to_global.copy()
            print(f"[RELOCALIZE] Matched keyframe {kf.id} (inliers={inliers})")
        else:
            # Truly lost - skip frame
            return
    
    # Step 3: Warp thermal to global canvas
    self._canvas_mgr.warp_and_blend(fp.thermal, fp.H_to_global)
    
    # Step 4: Add as keyframe if significant movement
    if self._should_add_keyframe(fp):
        self._add_keyframe(fp)
```

#### 4. **Relocalization Matcher**

```python
def _relocalize(self, rgb_curr: np.ndarray) -> Optional[KeyframeMatch]:
    """Match current frame against all keyframes."""
    kp_curr, des_curr = self._akaze.detectAndCompute(
        cv2.cvtColor(rgb_curr, cv2.COLOR_BGR2GRAY), None
    )
    
    best_match = None
    best_inliers = 0
    
    for kf in self._keyframes:
        matches = self._matcher.knnMatch(kf.descriptors, des_curr, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        if len(good) > MIN_RELOCALIZE_MATCHES:
            # Compute homography
            src = np.float32([kf.keypoints[m.queryIdx].pt for m in good])
            dst = np.float32([kp_curr[m.trainIdx].pt for m in good])
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            
            if H is not None:
                inliers = int(mask.sum())
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_match = KeyframeMatch(kf.id, H, inliers)
    
    return best_match if best_inliers > MIN_RELOCALIZE_INLIERS else None
```

#### 5. **Keyframe Selection Strategy**

Add keyframe when:
- **Distance:** Moved > 100px from last keyframe
- **Rotation:** Rotated > 10° from last keyframe
- **Time:** > 2 seconds since last keyframe
- **Coverage:** Viewing area with < 30% overlap with existing keyframes

```python
def _should_add_keyframe(self, fp: FramePair) -> bool:
    if not self._keyframes:
        return True
    
    last_kf = self._keyframes[-1]
    
    # Distance check
    dx = fp.H_to_global[0, 2] - last_kf.H_to_global[0, 2]
    dy = fp.H_to_global[1, 2] - last_kf.H_to_global[1, 2]
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist > 100:
        return True
    
    # Time check
    if fp.timestamp - last_kf.timestamp > 2.0:
        return True
    
    return False
```

---

## 📐 Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)

**Files to modify:**
- `panorama_stitcher.py`

**Changes:**
1. Add `Keyframe` dataclass
2. Add `_keyframes: list[Keyframe]` to `PanoramaStitcher`
3. Rename `H_acc` → `H_to_global` (semantic clarity)
4. Store features in `FramePair`: `features: Optional[tuple]`

### Phase 2: Relocalization System (3-4 hours)

**New methods:**
```python
def _relocalize(self, rgb: np.ndarray) -> Optional[KeyframeMatch]
def _add_keyframe(self, fp: FramePair) -> None
def _should_add_keyframe(self, fp: FramePair) -> bool
def _compute_overlap(self, fp: FramePair, kf: Keyframe) -> float
```

**Logic:**
- Extract features once per frame (reuse for matching)
- Match against all keyframes (can optimize with BoW later)
- Use RANSAC homography for robust matching
- Update `H_to_global` from best keyframe match

### Phase 3: Canvas Persistence (1-2 hours)

**Changes to `CanvasManager`:**
- Remove "override" semantics for revisited pixels
- **Keep soft blending** (70% new + 30% old) - already implemented ✅
- Add `get_thermal_at(x, y)` for querying existing data

### Phase 4: Drift Correction Enhancement (1 hour)

**Upgrade `_drift_correct()`:**
- Use keyframe database instead of just recent frames
- Perform loop closure when relocalization succeeds
- Adjust all keyframes in loop for global consistency

### Phase 5: Testing & Validation (2 hours)

**Test scenarios:**
1. ✅ Move camera left → right → left (return to start)
2. ✅ Scan monitor back → front → back
3. ✅ Circular motion (full 360° return)
4. ✅ Random walk with revisits

**Success criteria:**
- Relocalization triggers when returning to known area
- Thermal data persists in global coordinates
- No visible seams at revisited locations
- Keyframe count stays reasonable (< 100 for 5min scan)

---

## 🔧 Configuration Parameters

```python
@dataclass
class PanoramaConfig:
    # ... existing params ...
    
    # Relocalization
    min_relocalize_matches: int = 20        # Min feature matches to attempt relocalization
    min_relocalize_inliers: int = 15        # Min inliers to accept relocalization
    keyframe_distance_threshold: float = 100.0  # px
    keyframe_time_threshold: float = 2.0    # seconds
    keyframe_overlap_threshold: float = 0.7  # 70% overlap = don't add
    
    # Performance
    max_keyframes: int = 200                # Limit keyframe database size
    relocalize_every_n_frames: int = 5      # Only try relocalize every N frames
```

---

## 📊 Performance Considerations

### Memory Usage
- **Per keyframe:** ~2MB (640×480 RGB + thermal + features)
- **200 keyframes:** ~400MB
- **Mitigation:** Downsample RGB for feature extraction (320×240)

### CPU Usage
- **Feature extraction:** ~10ms per frame (AKAZE)
- **Relocalization:** ~50ms (match against 50 keyframes)
- **Mitigation:** Only relocalize when tracking lost

### Optimization Strategies
1. **Spatial indexing:** Grid-based keyframe lookup
2. **BoW (Bag of Words):** Fast keyframe retrieval
3. **Parallel matching:** Multi-thread keyframe matching
4. **Keyframe pruning:** Remove redundant keyframes

---

## 🎓 References & Inspiration

### Visual SLAM Systems
- **ORB-SLAM2:** Keyframe-based monocular SLAM
- **RTAB-Map:** Real-time appearance-based mapping
- **LSD-SLAM:** Direct image alignment

### Key Concepts Adapted
- ✅ Keyframe database
- ✅ Feature-based relocalization
- ✅ Global coordinate system
- ✅ Loop closure detection
- ❌ Bundle adjustment (too complex for 2D thermal)
- ❌ 3D reconstruction (not needed)

---

## 🚀 Expected Outcomes

### Before (Current System)
```
Camera path: A → B → A
Heat map:    [A] → [B] → [?]  ← A disappears!
```

### After (With Relocalization)
```
Camera path: A → B → A
Heat map:    [A] → [A+B] → [A+B]  ← A persists!
Keyframes:   KF1(A) → KF2(B) → relocalize→KF1
```

### User Experience
- ✅ Thermal data **never disappears**
- ✅ Can scan area in **any order**
- ✅ Can **revisit** locations to update thermal readings
- ✅ Heat map is **spatially consistent**
- ✅ Robot can move **freely** without losing data

---

## 📝 Next Steps

1. ✅ **Spec approved** - Review with team
2. ⏳ **Implement Phase 1** - Core infrastructure
3. ⏳ **Implement Phase 2** - Relocalization
4. ⏳ **Test & validate** - Real-world scenarios
5. ⏳ **Optimize** - Performance tuning
6. ⏳ **Deploy** - Production release

---

**Estimated Total Effort:** 9-12 hours  
**Complexity:** Complex (8/10)  
**Impact:** Critical - Fixes fundamental limitation  
**Risk:** Medium - Requires careful testing

