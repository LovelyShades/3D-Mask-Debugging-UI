// Apache 2.0 — based on MediaPipe Face Landmarker + TFJS FaceMesh demos.
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
import { TRIANGULATION } from "./triangulation.js";
import { UV_COORDS } from "./uv_coords.js"; // 468 items of [u,v] in [0..1]

const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const TEMPLATE_SRC = "./mesh_map.jpg";

const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");

let faceLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;
const videoWidth = 480;

// set true if your webcam/canvas is mirrored via CSS (e.g. rotateY(180deg))
const MIRROR_VIDEO = true;

// mask state
const maskInput = document.getElementById("maskInput");
const maskStatus = document.getElementById("maskStatus");
let maskImg = null;

// template state (fixed)
let templateImg = null;
let templateSize = { w: 0, h: 0 };
let srcPts = null;          // normal UV->px
let srcPtsFlipped = null;   // flipped horizontally

// ---------- load model ----------
(async () => {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1,
    });
    demosSection.classList.remove("invisible");
})();

// ---------- load fixed template (mesh_map.jpg) and precompute src points ----------
(function preloadTemplate() {
    const img = new Image();
    img.onload = () => {
        templateImg = img;
        templateSize = { w: img.naturalWidth, h: img.naturalHeight };

        // normal UVs
        srcPts = UV_COORDS.map(([u, v]) => ({
            x: u * templateSize.w,
            y: v * templateSize.h,
        }));
        // flipped horizontally (for mirrored video)
        srcPtsFlipped = UV_COORDS.map(([u, v]) => ({
            x: (1 - u) * templateSize.w,
            y: v * templateSize.h,
        }));
    };
    img.onerror = () => console.error("Failed to load template:", TEMPLATE_SRC);
    img.src = TEMPLATE_SRC;
})();

// ---------- mask upload ----------
if (maskInput) {
    maskInput.addEventListener("change", (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            maskImg = img;
            maskStatus.textContent = `Mask loaded (${img.naturalWidth}×${img.naturalHeight})`;
        };
        img.onerror = () => (maskStatus.textContent = "Failed to load PNG");
        img.src = url;
    });
}

// ---------- helpers ----------
function lmkPx(lmk, W, H) {
    return { x: lmk.x * W, y: lmk.y * H };
}

// Affine transform from triangle (src->dst) into canvas setTransform
function setTriTransform(ctx, src, dst) {
    const [s0, s1, s2] = src,
        [d0, d1, d2] = dst;
    const denom =
        s0.x * (s1.y - s2.y) +
        s1.x * (s2.y - s0.y) +
        s2.x * (s0.y - s1.y);
    if (Math.abs(denom) < 1e-8) return false;

    const a11 =
        (d0.x * (s1.y - s2.y) +
            d1.x * (s2.y - s0.y) +
            d2.x * (s0.y - s1.y)) /
        denom;
    const a12 =
        (d0.x * (s2.x - s1.x) +
            d1.x * (s0.x - s2.x) +
            d2.x * (s1.x - s0.x)) /
        denom;
    const a13 =
        (d0.x * (s1.x * s2.y - s2.x * s1.y) +
            d1.x * (s2.x * s0.y - s0.x * s2.y) +
            d2.x * (s0.x * s1.y - s1.x * s0.y)) /
        denom;

    const a21 =
        (d0.y * (s1.y - s2.y) +
            d1.y * (s2.y - s0.y) +
            d2.y * (s0.y - s1.y)) /
        denom;
    const a22 =
        (d0.y * (s2.x - s1.x) +
            d1.y * (s0.x - s2.x) +
            d2.y * (s1.x - s0.x)) /
        denom;
    const a23 =
        (d0.y * (s1.x * s2.y - s2.x * s1.y) +
            d1.y * (s2.x * s0.y - s0.x * s2.y) +
            d2.y * (s0.x * s1.y - s1.x * s0.y)) /
        denom;

    ctx.setTransform(a11, a21, a12, a22, a13, a23);
    return true;
}

// Warp the mask across the whole face using TRIANGULATION
function drawWarpedMask(ctx, dstLmks, useFlipped = false) {
    if (!maskImg || !srcPts || !dstLmks) return;

    const S = useFlipped ? srcPtsFlipped : srcPts;
    const W = ctx.canvas.width,
        H = ctx.canvas.height;

    for (let i = 0; i < TRIANGULATION.length; i += 3) {
        const i0 = TRIANGULATION[i],
            i1 = TRIANGULATION[i + 1],
            i2 = TRIANGULATION[i + 2];

        const s0 = S[i0],
            s1 = S[i1],
            s2 = S[i2];

        const d0 = lmkPx(dstLmks[i0], W, H);
        const d1 = lmkPx(dstLmks[i1], W, H);
        const d2 = lmkPx(dstLmks[i2], W, H);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(d0.x, d0.y);
        ctx.lineTo(d1.x, d1.y);
        ctx.lineTo(d2.x, d2.y);
        ctx.closePath();
        ctx.clip();

        if (setTriTransform(ctx, [s0, s1, s2], [d0, d1, d2])) {
            ctx.drawImage(maskImg, 0, 0, templateSize.w, templateSize.h);
        }
        ctx.restore();
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
}

/* -------------------------
   Demo 1: Click image detect
--------------------------*/
const imageContainers = document.getElementsByClassName("detectOnClick");
Array.from(imageContainers).forEach((container) => {
    const img = container.querySelector("img");
    if (img) img.addEventListener("click", handleClick);
});

async function handleClick(event) {
    if (!faceLandmarker) return;

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await faceLandmarker.setOptions({ runningMode });
    }

    const parent = event.target.parentNode;
    const allCanvas = parent.getElementsByClassName("canvas");
    for (let i = allCanvas.length - 1; i >= 0; i--) {
        allCanvas[i].parentNode.removeChild(allCanvas[i]);
    }

    const faceLandmarkerResult = faceLandmarker.detect(event.target);

    const canvas = document.createElement("canvas");
    canvas.className = "canvas";
    canvas.width = event.target.naturalWidth;
    canvas.height = event.target.naturalHeight;
    canvas.style.left = "0px";
    canvas.style.top = "0px";
    canvas.style.width = `${event.target.width}px`;
    canvas.style.height = `${event.target.height}px`;
    parent.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    const drawingUtils = new DrawingUtils(ctx);

    if (faceLandmarkerResult?.faceLandmarks) {
        for (const dstLmks of faceLandmarkerResult.faceLandmarks) {
            drawingUtils.drawConnectors(
                dstLmks,
                FaceLandmarker.FACE_LANDMARKS_TESSELATION,
                { color: "#C0C0C070", lineWidth: 1 }
            );
            // images are NOT mirrored -> use normal UVs
            drawWarpedMask(ctx, dstLmks, /*useFlipped=*/false);
        }
    }
    drawBlendShapes(imageBlendShapes, faceLandmarkerResult?.faceBlendshapes || []);
}

/* -------------------------
   Demo 2: Webcam detect
--------------------------*/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const videoDrawingUtils = new DrawingUtils(canvasCtx);

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    document.getElementById("webcamButton").addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

function enableCam() {
    if (!faceLandmarker) return;

    webcamRunning = !webcamRunning;
    document.getElementById("webcamButton").innerText = webcamRunning
        ? "DISABLE PREDICTIONS"
        : "ENABLE WEBCAM";

    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let lastVideoTime = -1;
let results;

async function predictWebcam() {
    const ratio = video.videoHeight / video.videoWidth;
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * ratio + "px";
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * ratio + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode });
    }

    const startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results?.faceLandmarks) {
        for (const dstLmks of results.faceLandmarks) {
            videoDrawingUtils.drawConnectors(
                dstLmks,
                FaceLandmarker.FACE_LANDMARKS_TESSELATION,
                { color: "#C0C0C070", lineWidth: 1 }
            );
            // webcam may be mirrored -> optionally use flipped UVs
            drawWarpedMask(canvasCtx, dstLmks, /*useFlipped=*/MIRROR_VIDEO);
        }
    }

    drawBlendShapes(videoBlendShapes, results?.faceBlendshapes || []);
    if (webcamRunning === true) window.requestAnimationFrame(predictWebcam);
}

function drawBlendShapes(el, blendShapes) {
    if (!blendShapes?.length) return;
    let html = "";
    blendShapes[0].categories.forEach((shape) => {
        const name = shape.displayName || shape.categoryName;
        const pct = Math.max(0, Math.min(100, Number(shape.score) * 100));
        html += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${name}</span>
        <span class="blend-shapes-value" style="width: calc(${pct}% - 120px)">
          ${Number(shape.score).toFixed(4)}
        </span>
      </li>
    `;
    });
    el.innerHTML = html;
}
