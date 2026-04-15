/**
 * planner.js — Canvas interaction and planning animation layer for the
 * MPPI Planner Visualizer.  Manages a Konva.js stage with six layers:
 *   1. Grid       – reference lines and labels
 *   2. SDF        – signed-distance-field heatmap (from server)
 *   3. Obstacle   – user-drawn rectangles and circles
 *   4. Path       – global path + MPPI trajectory polylines
 *   5. Pose       – start / goal markers with heading arrows
 *   6. Robot      – animated robot triangle during planning
 *
 * Exposes global functions consumed by the Alpine.js UI in index.html:
 *   initCanvas, startPlanning, clearCanvas, getObstacles, getPoses,
 *   setDrawMode
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // World / canvas coordinate system
  // ---------------------------------------------------------------------------

  let worldMinX = -3;
  let worldMaxX = 3;
  let worldMinY = -3;
  let worldMaxY = 3;
  let scale = 100; // pixels per meter — recomputed on resize

  function worldToCanvas(wx, wy) {
    return {
      x: (wx - worldMinX) * scale,
      y: (worldMaxY - wy) * scale,
    };
  }

  function canvasToWorld(cx, cy) {
    return {
      x: cx / scale + worldMinX,
      y: worldMaxY - cy / scale,
    };
  }

  // ---------------------------------------------------------------------------
  // Module state
  // ---------------------------------------------------------------------------

  let stage = null;
  let gridLayer = null;
  let sdfLayer = null;
  let obstacleLayer = null;
  let pathLayer = null;
  let poseLayer = null;
  let robotLayer = null;

  let drawMode = "rect"; // 'rect' | 'circle' | 'start' | 'goal' | 'erase'

  // Stored obstacle definitions in *world* coords
  // { type: 'rect', x1, y1, x2, y2, shape: Konva.Rect }
  // { type: 'circle', cx, cy, r, shape: Konva.Circle }
  const obstacles = [];

  // Current poses (world coords + heading in radians)
  let startPose = null; // { x, y, heading, shape }
  let goalPose = null;

  // Drawing transient state
  let isDrawing = false;
  let drawOrigin = null; // { cx, cy } canvas coords at mousedown
  let previewShape = null;
  let headingLine = null; // line showing drag direction for pose heading

  // Active EventSource (so we can close it on clear / re-plan)
  let activeEventSource = null;

  // Stored trajectory for playback (filled during planning, replayed on Play)
  let storedSteps = []; // array of { state, action, pos_error, heading_error, predicted_trajectory }
  let storedSmoothedTraj = null; // SGF-smoothed baseline from backend

  // ---------------------------------------------------------------------------
  // Grid rendering
  // ---------------------------------------------------------------------------

  function buildGrid() {
    gridLayer.destroyChildren();

    const worldW = worldMaxX - worldMinX;
    const worldH = worldMaxY - worldMinY;
    const canvasW = worldW * scale;
    const canvasH = worldH * scale;

    // Minor grid lines every 0.5 m
    const step = 0.5;
    for (let wx = worldMinX; wx <= worldMaxX; wx += step) {
      const cx = (wx - worldMinX) * scale;
      const isAxis = Math.abs(wx) < 1e-6;
      gridLayer.add(
        new Konva.Line({
          points: [cx, 0, cx, canvasH],
          stroke: isAxis ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.1)",
          strokeWidth: isAxis ? 1.5 : 0.5,
          listening: false,
        })
      );
    }
    for (let wy = worldMinY; wy <= worldMaxY; wy += step) {
      const cy = (worldMaxY - wy) * scale;
      const isAxis = Math.abs(wy) < 1e-6;
      gridLayer.add(
        new Konva.Line({
          points: [0, cy, canvasW, cy],
          stroke: isAxis ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.1)",
          strokeWidth: isAxis ? 1.5 : 0.5,
          listening: false,
        })
      );
    }

    // Coordinate labels every 1 m along the edges
    const fontSize = Math.max(9, Math.round(scale * 0.1));
    for (let wx = Math.ceil(worldMinX); wx <= Math.floor(worldMaxX); wx += 1) {
      const cx = (wx - worldMinX) * scale;
      gridLayer.add(
        new Konva.Text({
          x: cx + 2,
          y: canvasH - fontSize - 2,
          text: String(wx),
          fontSize: fontSize,
          fill: "rgba(255,255,255,0.4)",
          listening: false,
        })
      );
    }
    for (let wy = Math.ceil(worldMinY); wy <= Math.floor(worldMaxY); wy += 1) {
      const cy = (worldMaxY - wy) * scale;
      gridLayer.add(
        new Konva.Text({
          x: 2,
          y: cy + 2,
          text: String(wy),
          fontSize: fontSize,
          fill: "rgba(255,255,255,0.4)",
          listening: false,
        })
      );
    }

    gridLayer.batchDraw();
  }

  // ---------------------------------------------------------------------------
  // Pose marker helpers
  // ---------------------------------------------------------------------------

  function createPoseMarker(wx, wy, heading, color) {
    const pos = worldToCanvas(wx, wy);
    const r = Math.max(8, scale * 0.08);
    const group = new Konva.Group({ x: pos.x, y: pos.y });

    // Triangle pointing right (heading=0), then rotated
    const triangle = new Konva.RegularPolygon({
      sides: 3,
      radius: r,
      fill: color,
      stroke: "#fff",
      strokeWidth: 1.5,
      opacity: 0.9,
    });
    group.add(triangle);

    // Heading line (short arrow extension)
    const lineLen = r * 1.8;
    const hLine = new Konva.Line({
      points: [0, 0, lineLen, 0],
      stroke: "#fff",
      strokeWidth: 1.5,
      opacity: 0.7,
    });
    group.add(hLine);

    // Rotation: Konva uses degrees, CW.  World heading is CCW from +X.
    // Canvas Y is flipped, so canvas angle = -worldHeading.
    group.rotation(-heading * (180 / Math.PI));

    return group;
  }

  function placeOrReplacePose(which, wx, wy, heading) {
    const color = which === "start" ? "#22c55e" : "#ef4444";
    const marker = createPoseMarker(wx, wy, heading, color);
    poseLayer.add(marker);

    if (which === "start") {
      if (startPose && startPose.shape) startPose.shape.destroy();
      startPose = { x: wx, y: wy, heading: heading, shape: marker };
    } else {
      if (goalPose && goalPose.shape) goalPose.shape.destroy();
      goalPose = { x: wx, y: wy, heading: heading, shape: marker };
    }
    poseLayer.batchDraw();
  }

  // ---------------------------------------------------------------------------
  // Obstacle creation helpers
  // ---------------------------------------------------------------------------

  function addRectObstacle(cx1, cy1, cx2, cy2) {
    // Normalise so (cx1,cy1) is top-left in canvas space
    const x = Math.min(cx1, cx2);
    const y = Math.min(cy1, cy2);
    const w = Math.abs(cx2 - cx1);
    const h = Math.abs(cy2 - cy1);

    // Check minimum world size
    if (w / scale < 0.1 && h / scale < 0.1) return;

    const rect = new Konva.Rect({
      x: x,
      y: y,
      width: w,
      height: h,
      fill: "rgba(59,130,246,0.3)",
      stroke: "#3b82f6",
      strokeWidth: 1.5,
      draggable: true,
      name: "obstacle",
    });
    obstacleLayer.add(rect);

    const wTL = canvasToWorld(x, y);
    const wBR = canvasToWorld(x + w, y + h);
    const entry = {
      type: "rect",
      x1: Math.min(wTL.x, wBR.x),
      y1: Math.min(wTL.y, wBR.y),
      x2: Math.max(wTL.x, wBR.x),
      y2: Math.max(wTL.y, wBR.y),
      shape: rect,
    };
    obstacles.push(entry);

    // Keep world coords in sync when dragged
    rect.on("dragend", function () {
      const nx = rect.x();
      const ny = rect.y();
      const nw = rect.width();
      const nh = rect.height();
      const tl = canvasToWorld(nx, ny);
      const br = canvasToWorld(nx + nw, ny + nh);
      entry.x1 = Math.min(tl.x, br.x);
      entry.y1 = Math.min(tl.y, br.y);
      entry.x2 = Math.max(tl.x, br.x);
      entry.y2 = Math.max(tl.y, br.y);
    });

    // Double-click to delete
    rect.on("dblclick dbltap", function () {
      removeObstacleEntry(entry);
    });

    obstacleLayer.batchDraw();
  }

  function addCircleObstacle(ccx, ccy, cr) {
    if (cr / scale < 0.05) return;

    const circle = new Konva.Circle({
      x: ccx,
      y: ccy,
      radius: cr,
      fill: "rgba(59,130,246,0.3)",
      stroke: "#3b82f6",
      strokeWidth: 1.5,
      draggable: true,
      name: "obstacle",
    });
    obstacleLayer.add(circle);

    const wc = canvasToWorld(ccx, ccy);
    const entry = {
      type: "circle",
      cx: wc.x,
      cy: wc.y,
      r: cr / scale,
      shape: circle,
    };
    obstacles.push(entry);

    circle.on("dragend", function () {
      const wNew = canvasToWorld(circle.x(), circle.y());
      entry.cx = wNew.x;
      entry.cy = wNew.y;
    });

    circle.on("dblclick dbltap", function () {
      removeObstacleEntry(entry);
    });

    obstacleLayer.batchDraw();
  }

  function removeObstacleEntry(entry) {
    entry.shape.destroy();
    const idx = obstacles.indexOf(entry);
    if (idx !== -1) obstacles.splice(idx, 1);
    obstacleLayer.batchDraw();
  }

  // ---------------------------------------------------------------------------
  // SDF rendering
  // ---------------------------------------------------------------------------

  function renderSDF(sdfGrid, origin, resolution) {
    const W = sdfGrid.length;
    if (W === 0) return;
    const H = sdfGrid[0].length;

    const offscreen = document.createElement("canvas");
    offscreen.width = W;
    offscreen.height = H;
    const ctx = offscreen.getContext("2d");
    const imgData = ctx.createImageData(W, H);

    for (let i = 0; i < W; i++) {
      for (let j = 0; j < H; j++) {
        const val = sdfGrid[i][j];
        // Flip j for canvas Y direction
        const pixIdx = ((H - 1 - j) * W + i) * 4;

        if (val < 0) {
          // Inside obstacle: red
          imgData.data[pixIdx] = 220;
          imgData.data[pixIdx + 1] = 50;
          imgData.data[pixIdx + 2] = 50;
          imgData.data[pixIdx + 3] = 180;
        } else if (val < 0.2) {
          // Near obstacle surface: yellow/orange fade
          const t = val / 0.2;
          imgData.data[pixIdx] = 220;
          imgData.data[pixIdx + 1] = Math.floor(50 + 170 * t);
          imgData.data[pixIdx + 2] = 50;
          imgData.data[pixIdx + 3] = Math.floor(180 * (1 - t * 0.7));
        } else {
          // Free space: transparent
          imgData.data[pixIdx] = 0;
          imgData.data[pixIdx + 1] = 0;
          imgData.data[pixIdx + 2] = 0;
          imgData.data[pixIdx + 3] = 0;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Scale the SDF image to match world extents on the Konva canvas
    const worldW = W * resolution;
    const worldH = H * resolution;
    const canvasPos = worldToCanvas(origin[0], origin[1] + worldH);

    const sdfImage = new Konva.Image({
      image: offscreen,
      x: canvasPos.x,
      y: canvasPos.y,
      width: worldW * scale,
      height: worldH * scale,
      listening: false,
    });

    sdfLayer.destroyChildren();
    sdfLayer.add(sdfImage);
    sdfLayer.batchDraw();
  }

  // ---------------------------------------------------------------------------
  // Stage mouse/touch interaction
  // ---------------------------------------------------------------------------

  function attachDrawingHandlers() {
    stage.on("mousedown touchstart", function (e) {
      // Ignore if the event target is an obstacle shape (for dragging / erase)
      if (drawMode === "erase") {
        const target = e.target;
        if (target && target.name && target.name() === "obstacle") {
          // Find the matching entry and remove it
          const entry = obstacles.find(function (o) {
            return o.shape === target;
          });
          if (entry) removeObstacleEntry(entry);
        }
        return;
      }

      // Don't start drawing if we clicked on an existing obstacle (let drag work)
      if (
        e.target !== stage &&
        e.target.name &&
        e.target.name() === "obstacle"
      ) {
        return;
      }

      const pointer = stage.getPointerPosition();
      if (!pointer) return;

      isDrawing = true;
      drawOrigin = { cx: pointer.x, cy: pointer.y };

      if (drawMode === "rect") {
        previewShape = new Konva.Rect({
          x: pointer.x,
          y: pointer.y,
          width: 0,
          height: 0,
          stroke: "#3b82f6",
          strokeWidth: 1.5,
          dash: [6, 3],
          listening: false,
        });
        obstacleLayer.add(previewShape);
      } else if (drawMode === "circle") {
        previewShape = new Konva.Circle({
          x: pointer.x,
          y: pointer.y,
          radius: 0,
          stroke: "#3b82f6",
          strokeWidth: 1.5,
          dash: [6, 3],
          listening: false,
        });
        obstacleLayer.add(previewShape);
      } else if (drawMode === "start" || drawMode === "goal") {
        headingLine = new Konva.Line({
          points: [pointer.x, pointer.y, pointer.x, pointer.y],
          stroke: drawMode === "start" ? "#22c55e" : "#ef4444",
          strokeWidth: 2,
          dash: [4, 4],
          listening: false,
        });
        poseLayer.add(headingLine);
      }
    });

    stage.on("mousemove touchmove", function () {
      if (!isDrawing || !drawOrigin) return;
      const pointer = stage.getPointerPosition();
      if (!pointer) return;

      if (drawMode === "rect" && previewShape) {
        const x = Math.min(drawOrigin.cx, pointer.x);
        const y = Math.min(drawOrigin.cy, pointer.y);
        const w = Math.abs(pointer.x - drawOrigin.cx);
        const h = Math.abs(pointer.y - drawOrigin.cy);
        previewShape.setAttrs({ x: x, y: y, width: w, height: h });
        obstacleLayer.batchDraw();
      } else if (drawMode === "circle" && previewShape) {
        const dx = pointer.x - drawOrigin.cx;
        const dy = pointer.y - drawOrigin.cy;
        previewShape.radius(Math.sqrt(dx * dx + dy * dy));
        obstacleLayer.batchDraw();
      } else if (
        (drawMode === "start" || drawMode === "goal") &&
        headingLine
      ) {
        headingLine.points([
          drawOrigin.cx,
          drawOrigin.cy,
          pointer.x,
          pointer.y,
        ]);
        poseLayer.batchDraw();
      }
    });

    stage.on("mouseup touchend", function () {
      if (!isDrawing || !drawOrigin) return;
      const pointer = stage.getPointerPosition();
      isDrawing = false;

      if (drawMode === "rect" && previewShape) {
        const cx1 = drawOrigin.cx;
        const cy1 = drawOrigin.cy;
        const cx2 = pointer ? pointer.x : cx1;
        const cy2 = pointer ? pointer.y : cy1;
        previewShape.destroy();
        previewShape = null;
        addRectObstacle(cx1, cy1, cx2, cy2);
      } else if (drawMode === "circle" && previewShape) {
        const ccx = drawOrigin.cx;
        const ccy = drawOrigin.cy;
        const dx = (pointer ? pointer.x : ccx) - ccx;
        const dy = (pointer ? pointer.y : ccy) - ccy;
        const cr = Math.sqrt(dx * dx + dy * dy);
        previewShape.destroy();
        previewShape = null;
        addCircleObstacle(ccx, ccy, cr);
      } else if (drawMode === "start" || drawMode === "goal") {
        if (headingLine) {
          headingLine.destroy();
          headingLine = null;
        }

        const endX = pointer ? pointer.x : drawOrigin.cx;
        const endY = pointer ? pointer.y : drawOrigin.cy;
        const dragDx = endX - drawOrigin.cx;
        const dragDy = endY - drawOrigin.cy;

        const wPos = canvasToWorld(drawOrigin.cx, drawOrigin.cy);

        let heading = 0;
        if (Math.sqrt(dragDx * dragDx + dragDy * dragDy) >= 5) {
          // Compute heading in world space.
          // Canvas drag vector: (dragDx, dragDy).  In world space Y is flipped.
          heading = Math.atan2(-dragDy, dragDx);
        }

        placeOrReplacePose(drawMode, wPos.x, wPos.y, heading);
        poseLayer.batchDraw();
      }

      drawOrigin = null;
    });
  }

  // ---------------------------------------------------------------------------
  // Resize handling
  // ---------------------------------------------------------------------------

  function fitStage() {
    const container = document.getElementById("konva-container");
    if (!container || !stage) return;

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw === 0 || ch === 0) return;

    // Use a square canvas region based on the smaller dimension
    const size = Math.min(cw, ch);
    const worldSpan = worldMaxX - worldMinX; // assumed equal in X and Y
    scale = size / worldSpan;

    stage.width(size);
    stage.height(size);

    // Rebuild grid with new scale
    buildGrid();

    // Reposition existing obstacle shapes
    obstacles.forEach(function (entry) {
      if (entry.type === "rect") {
        const tl = worldToCanvas(entry.x1, entry.y2);
        const br = worldToCanvas(entry.x2, entry.y1);
        entry.shape.setAttrs({
          x: tl.x,
          y: tl.y,
          width: br.x - tl.x,
          height: br.y - tl.y,
        });
      } else if (entry.type === "circle") {
        const cp = worldToCanvas(entry.cx, entry.cy);
        entry.shape.setAttrs({
          x: cp.x,
          y: cp.y,
          radius: entry.r * scale,
        });
      }
    });
    obstacleLayer.batchDraw();

    // Reposition pose markers
    if (startPose) {
      placeOrReplacePose("start", startPose.x, startPose.y, startPose.heading);
    }
    if (goalPose) {
      placeOrReplacePose("goal", goalPose.x, goalPose.y, goalPose.heading);
    }
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  window.initCanvas = function () {
    const container = document.getElementById("konva-container");
    if (!container) {
      console.error("planner.js: #konva-container not found");
      return;
    }

    const size = Math.min(container.clientWidth, container.clientHeight) || 600;
    scale = size / (worldMaxX - worldMinX);

    stage = new Konva.Stage({
      container: "konva-container",
      width: size,
      height: size,
    });

    // Create layers bottom-to-top
    gridLayer = new Konva.Layer({ listening: false });
    sdfLayer = new Konva.Layer({ listening: false });
    obstacleLayer = new Konva.Layer();
    pathLayer = new Konva.Layer({ listening: false });
    poseLayer = new Konva.Layer();
    robotLayer = new Konva.Layer({ listening: false });

    stage.add(gridLayer);
    stage.add(sdfLayer);
    stage.add(obstacleLayer);
    stage.add(pathLayer);
    stage.add(poseLayer);
    stage.add(robotLayer);

    buildGrid();
    attachDrawingHandlers();

    // Resize observer
    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(function () {
        fitStage();
      });
      ro.observe(container);
    }
  };

  window.startPlanning = function (url) {
    // Close previous stream if still open
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }

    // Append obstacle, pose, and bounds data as query params matching the backend
    const obsData = window.getObstacles();
    const poseData = window.getPoses();
    const boundsArr = [worldMinX, worldMinY, worldMaxX, worldMaxY];
    const sep = url.indexOf("?") === -1 ? "?" : "&";
    const fullUrl =
      url +
      sep +
      "boxes=" + encodeURIComponent(JSON.stringify(obsData.boxes)) +
      "&circles=" + encodeURIComponent(JSON.stringify(obsData.circles)) +
      "&start=" + encodeURIComponent(JSON.stringify(poseData.start || [0, 0, 0])) +
      "&goal=" + encodeURIComponent(JSON.stringify(poseData.goal || [2, 2, 0])) +
      "&bounds=" + encodeURIComponent(JSON.stringify(boundsArr));

    // Clear previous visualization
    pathLayer.destroyChildren();
    robotLayer.destroyChildren();
    sdfLayer.destroyChildren();
    pathLayer.batchDraw();
    robotLayer.batchDraw();
    sdfLayer.batchDraw();
    storedSteps = [];

    return new Promise(function (resolve, reject) {
      const evtSource = new EventSource(fullUrl);
      activeEventSource = evtSource;

      // -- SDF event --------------------------------------------------------
      evtSource.addEventListener("sdf", function (e) {
        try {
          const data = JSON.parse(e.data);
          renderSDF(data.grid, data.origin, data.resolution);
        } catch (err) {
          console.warn("planner.js: bad sdf event", err);
        }
      });

      // -- Global path event ------------------------------------------------
      evtSource.addEventListener("global_path", function (e) {
        try {
          const data = JSON.parse(e.data);
          const pts = [];
          data.path.forEach(function (p) {
            const c = worldToCanvas(p[0], p[1]);
            pts.push(c.x, c.y);
          });
          const line = new Konva.Line({
            points: pts,
            stroke: "#22c55e",
            strokeWidth: 2,
            dash: [8, 4],
            opacity: 0.7,
            listening: false,
          });
          pathLayer.add(line);
          // Draw heading arrows at each RRT waypoint
          var arrowLen = Math.max(10, scale * 0.1);
          data.path.forEach(function (p) {
            var c = worldToCanvas(p[0], p[1]);
            var heading = p[2] || 0;
            // Arrow tip direction in canvas coords (Y flipped)
            var dx = Math.cos(heading) * arrowLen;
            var dy = -Math.sin(heading) * arrowLen;
            pathLayer.add(new Konva.Arrow({
              x: c.x,
              y: c.y,
              points: [0, 0, dx, dy],
              pointerLength: 5,
              pointerWidth: 5,
              fill: "#22c55e",
              stroke: "#22c55e",
              strokeWidth: 2,
              opacity: 0.85,
              listening: false,
            }));
            // Small dot at the base
            pathLayer.add(new Konva.Circle({
              x: c.x,
              y: c.y,
              radius: 2.5,
              fill: "#fff",
              listening: false,
              opacity: 0.9,
            }));
          });
          pathLayer.batchDraw();
        } catch (err) {
          console.warn("planner.js: bad global_path event", err);
        }
      });

      // -- Step event: store silently (no animation yet) --------------------
      evtSource.addEventListener("step", function (e) {
        try {
          const data = JSON.parse(e.data);
          storedSteps.push(data);
        } catch (err) {
          console.warn("planner.js: bad step event", err);
        }
      });

      // -- Done event -------------------------------------------------------
      evtSource.addEventListener("done", function (e) {
        let data = {};
        try {
          data = JSON.parse(e.data);
        } catch (_) {}
        evtSource.close();
        activeEventSource = null;
        data.steps = storedSteps.length;
        // Store SGF-smoothed baseline
        storedSmoothedTraj = data.smoothed_trajectory || null;
        window.dispatchEvent(
          new CustomEvent("planner-done", { detail: data })
        );
        resolve(data);
      });

      // -- Error handling ---------------------------------------------------
      evtSource.addEventListener("error", function (e) {
        let msg = "Connection error";
        try { msg = JSON.parse(e.data).message; } catch (_) {}
        evtSource.close();
        activeEventSource = null;
        window.dispatchEvent(
          new CustomEvent("planner-done", { detail: { status: "Error", message: msg } })
        );
        resolve({ status: "Error" });
      });

      evtSource.onerror = function () {
        evtSource.close();
        activeEventSource = null;
        window.dispatchEvent(
          new CustomEvent("planner-done", { detail: { status: "Error" } })
        );
        resolve({ status: "Error" });
      };
    });
  };

  /**
   * Animate the stored trajectory step-by-step.
   * Called after planning completes, when user clicks "Play".
   */
  window.playTrajectory = function () {
    if (storedSteps.length === 0) return Promise.resolve();

    // Clear previous playback visuals
    robotLayer.destroyChildren();
    var oldTraj = pathLayer.findOne(".traj-line");
    if (oldTraj) oldTraj.destroy();
    var oldHorizon = pathLayer.findOne(".horizon-line");
    if (oldHorizon) oldHorizon.destroy();
    var oldSmoothed = pathLayer.findOne(".smoothed-line");
    if (oldSmoothed) oldSmoothed.destroy();
    pathLayer.batchDraw();

    var robot = new Konva.RegularPolygon({
      sides: 3,
      radius: Math.max(8, scale * 0.08),
      fill: "white",
      stroke: "#333",
      strokeWidth: 1,
    });
    robotLayer.add(robot);

    // Executed trajectory (blue)
    var trajPoints = [];
    var trajLine = new Konva.Line({
      points: [],
      stroke: "#38bdf8",
      strokeWidth: 2.5,
      lineCap: "round",
      lineJoin: "round",
      listening: false,
      name: "traj-line",
    });
    pathLayer.add(trajLine);

    // Predicted horizon line (thin cyan, redrawn each step)
    var horizonLine = new Konva.Line({
      points: [],
      stroke: "#06b6d4",
      strokeWidth: 1.5,
      opacity: 0.6,
      dash: [4, 3],
      listening: false,
      name: "horizon-line",
    });
    pathLayer.add(horizonLine);

    // Sampled rollout lines: green (best) → red (worst), evenly spaced by cost
    var sampleLines = [];
    var N_SAMPLES = 10;
    for (var si = 0; si < N_SAMPLES; si++) {
      var t = si / (N_SAMPLES - 1); // 0=best, 1=worst
      var r = Math.round(80 + 175 * t);
      var g = Math.round(220 - 170 * t);
      var b = Math.round(80 - 30 * t);
      var a = (0.4 - 0.25 * t).toFixed(2);
      var sl = new Konva.Line({
        points: [],
        stroke: "rgba(" + r + "," + g + "," + b + "," + a + ")",
        strokeWidth: 1.5 - t * 0.8,
        listening: false,
        name: "sample-line",
      });
      pathLayer.add(sl);
      sampleLines.push(sl);
    }

    var stepIdx = 0;
    var FRAME_MS = 30;

    return new Promise(function (resolve) {
      function tick() {
        if (stepIdx >= storedSteps.length) {
          // Remove horizon and sample lines at end
          horizonLine.points([]);
          sampleLines.forEach(function (sl) { sl.points([]); });

          // Draw SGF-smoothed baseline (orange dashed)
          if (storedSmoothedTraj && storedSmoothedTraj.length > 1) {
            var smoothPts = [];
            storedSmoothedTraj.forEach(function (s) {
              var c = worldToCanvas(s[0], s[1]);
              smoothPts.push(c.x, c.y);
            });
            var smoothLine = new Konva.Line({
              points: smoothPts,
              stroke: "#f97316",
              strokeWidth: 2,
              dash: [6, 4],
              opacity: 0.8,
              listening: false,
              name: "smoothed-line",
            });
            pathLayer.add(smoothLine);
          }

          robotLayer.batchDraw();
          pathLayer.batchDraw();
          window.dispatchEvent(new CustomEvent("playback-done"));
          resolve();
          return;
        }

        var data = storedSteps[stepIdx];
        var pos = worldToCanvas(data.state[0], data.state[1]);

        // Move robot
        robot.position(pos);
        robot.rotation(-data.state[2] * (180 / Math.PI) + 90);

        // Extend executed trajectory
        trajPoints.push(pos.x, pos.y);
        trajLine.points(trajPoints.slice());

        // Draw predicted horizon (MPPI lookahead)
        if (data.predicted_trajectory && data.predicted_trajectory.length > 1) {
          var hPts = [];
          data.predicted_trajectory.forEach(function (s) {
            var c = worldToCanvas(s[0], s[1]);
            hPts.push(c.x, c.y);
          });
          horizonLine.points(hPts);
        }

        // Draw sampled rollout trajectories (top-K by weight)
        if (data.sampled_trajectories) {
          for (var si = 0; si < sampleLines.length; si++) {
            if (si < data.sampled_trajectories.length) {
              var sPts = [];
              data.sampled_trajectories[si].forEach(function (p) {
                var c = worldToCanvas(p[0], p[1]);
                sPts.push(c.x, c.y);
              });
              sampleLines[si].points(sPts);
            } else {
              sampleLines[si].points([]);
            }
          }
        }

        robotLayer.batchDraw();
        pathLayer.batchDraw();

        // Dispatch step event for charts/status
        window.dispatchEvent(
          new CustomEvent("playback-step", {
            detail: {
              step: data.step,
              pos_error: data.pos_error,
              heading_error: data.heading_error,
              action: data.action,
              goal_reward_triggered: data.goal_reward_triggered || false,
            },
          })
        );

        stepIdx++;
        setTimeout(tick, FRAME_MS);
      }
      tick();
    });
  };

  window.clearCanvas = function () {
    // Close any active planning stream
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }

    // Clear stored trajectory
    storedSteps = [];
    storedSmoothedTraj = null;

    // Remove all obstacles
    obstacles.length = 0;
    obstacleLayer.destroyChildren();
    obstacleLayer.batchDraw();

    // Remove paths
    pathLayer.destroyChildren();
    pathLayer.batchDraw();

    // Remove SDF
    sdfLayer.destroyChildren();
    sdfLayer.batchDraw();

    // Remove robot
    robotLayer.destroyChildren();
    robotLayer.batchDraw();

    // Remove poses
    if (startPose && startPose.shape) startPose.shape.destroy();
    if (goalPose && goalPose.shape) goalPose.shape.destroy();
    startPose = null;
    goalPose = null;
    poseLayer.destroyChildren();
    poseLayer.batchDraw();
  };

  window.getObstacles = function () {
    const boxes = [];
    const circles = [];
    obstacles.forEach(function (o) {
      if (o.type === "rect") {
        boxes.push([o.x1, o.y1, o.x2, o.y2]);
      } else if (o.type === "circle") {
        circles.push([o.cx, o.cy, o.r]);
      }
    });
    return { boxes: boxes, circles: circles };
  };

  window.getPoses = function () {
    return {
      start: startPose ? [startPose.x, startPose.y, startPose.heading] : null,
      goal: goalPose ? [goalPose.x, goalPose.y, goalPose.heading] : null,
    };
  };

  /**
   * Load a preset scene: clear canvas, add obstacles and poses from world coords.
   * @param {Object} preset - { boxes: [[x1,y1,x2,y2],...], circles: [[cx,cy,r],...], start: [x,y,h], goal: [x,y,h] }
   */
  window.loadPreset = function (preset) {
    window.clearCanvas();

    // Add box obstacles
    (preset.boxes || []).forEach(function (b) {
      var tl = worldToCanvas(b[0], b[3]);
      var br = worldToCanvas(b[2], b[1]);
      addRectObstacle(tl.x, tl.y, br.x, br.y);
    });

    // Add circle obstacles
    (preset.circles || []).forEach(function (c) {
      var cp = worldToCanvas(c[0], c[1]);
      addCircleObstacle(cp.x, cp.y, c[2] * scale);
    });

    // Place poses
    if (preset.start) {
      placeOrReplacePose("start", preset.start[0], preset.start[1], preset.start[2]);
    }
    if (preset.goal) {
      placeOrReplacePose("goal", preset.goal[0], preset.goal[1], preset.goal[2]);
    }
  };

  window.setDrawMode = function (mode) {
    drawMode = mode;

    // Toggle draggable on obstacle shapes depending on erase mode
    const isDraggable = mode !== "erase";
    obstacles.forEach(function (o) {
      o.shape.draggable(isDraggable);
    });
  };

  // Keep drawMode in sync when Alpine.js changes it via the toolbar buttons.
  // Alpine sets `drawMode` on its own scope, so we watch for changes via a
  // MutationObserver on the button classes (lightweight approach) or simply
  // expose setDrawMode and have Alpine call it.  The index.html toolbar
  // buttons already bind @click="drawMode = '...'" on the Alpine scope, so
  // we bridge by watching a global.  An easy method: Alpine dispatches a
  // custom event, or we poll.  Simplest: use a Proxy on window or just rely
  // on Alpine calling setDrawMode.  We'll use an interval-free approach:
  // override Alpine's drawMode reactivity by adding x-effect in index.html
  // OR simply watching the Alpine store.
  //
  // For now, we keep it simple: the toolbar buttons set Alpine's drawMode,
  // and our mousedown reads from the module-level drawMode.  We synchronise
  // by listening for a custom event that index.html can dispatch, or by
  // reading from the Alpine component directly.

  // Fallback sync: poll-free bridge using requestAnimationFrame
  (function syncDrawMode() {
    // Attempt to read Alpine's drawMode from the main element
    const mainEl = document.querySelector("main[x-data]");
    if (mainEl && mainEl.__x) {
      const alpineMode = mainEl.__x.$data.drawMode;
      if (alpineMode && alpineMode !== drawMode) {
        window.setDrawMode(alpineMode);
      }
    } else if (mainEl && mainEl._x_dataStack) {
      // Alpine v3 internal
      try {
        const stack = mainEl._x_dataStack;
        if (stack && stack.length > 0) {
          const data = stack[0];
          if (data.drawMode && data.drawMode !== drawMode) {
            window.setDrawMode(data.drawMode);
          }
        }
      } catch (_) {
        // ignore
      }
    }
    requestAnimationFrame(syncDrawMode);
  })();
})();
