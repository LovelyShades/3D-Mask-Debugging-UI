// Apache 2.0 — based on MediaPipe Face Landmarker demos.
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
import { TRIANGULATION } from "./triangulation.js";
import { UV_COORDS } from "./uv_coords.js"; // 468 items of [u,v] in [0..1]
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js";

const { FaceLandmarker, FilesetResolver } = vision;

const TEMPLATE_SRC = "TESTING_IMAGES/mesh_map.jpg"; // optional
const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");

let faceLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;
const videoWidth = 480;

// ---- flip controls ----
const MIRROR_VIDEO = false;   // your webcam is not mirrored in CSS
const MIRROR_IMAGE = false;   // sample image not mirrored

// mask state + UI
const maskInput = document.getElementById("maskInput");
const maskStatus = document.getElementById("maskStatus");
const clearMaskBtnMaybe = document.getElementById("clearMaskBtn"); // may or may not exist in your HTML
let clearBtn = clearMaskBtnMaybe || null;
let maskImg = null;

// If the Clear button isn't present in HTML, create it next to the input (keeps your CSS working)
if (!clearBtn && maskInput) {
    clearBtn = document.createElement("button");
    clearBtn.id = "clearMaskBtn";
    clearBtn.className = "mdc-button mdc-button--outlined";
    clearBtn.style.display = "none";
    clearBtn.style.marginLeft = "8px";
    clearBtn.innerHTML = `<span class="mdc-button__ripple"></span><span class="mdc-button__label">Clear Mask</span>`;
    maskInput.insertAdjacentElement("afterend", clearBtn);
}
// Wire it: reload the page when clearing
if (clearBtn) {
    clearBtn.addEventListener("click", () => window.location.reload());
}

// template (optional)
let templateSize = { w: 0, h: 0 };

// UI: mesh toggle -> wireframe UNDER the mask
let showMesh = true;
document.getElementById("meshToggleBtn").addEventListener("click", () => {
    showMesh = !showMesh;
    document.getElementById("meshToggleBtn")
        .querySelector(".mdc-button__label").textContent = showMesh ? "Hide Mesh" : "Show Mesh";
    if (wireMesh) wireMesh.visible = showMesh; // webcam mesh updates live
});

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

// ---------- optional template ----------
(function preloadTemplate() {
    const img = new Image();
    img.onload = () => (templateSize = { w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => console.warn("Failed to load template:", TEMPLATE_SRC);
    img.src = TEMPLATE_SRC;
})();

// ---------- mask upload ----------
let maskTexture = null; // shared for webcam
if (maskInput) {
    maskInput.addEventListener("change", (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            maskImg = img;
            maskStatus.textContent = `Mask loaded (${img.naturalWidth}×${img.naturalHeight})`;

            // Refresh webcam texture immediately if running/initialized
            if (maskTexture) {
                maskTexture.image = maskImg;
                maskTexture.wrapS = THREE.RepeatWrapping;
                maskTexture.repeat.x = -1; // horizontal flip so text reads correctly in webcam
                maskTexture.offset.x = 1;
                maskTexture.needsUpdate = true;
            }
            // If a webcam mesh exists, make sure it becomes visible
            if (mesh) mesh.visible = true;

            // Clean old image overlays so the next click is clean
            document.querySelectorAll(".detectOnClick .canvas").forEach((c) => c.remove());

            // ---- Turn the upload control into "Clear mask" in-place ----
            const labelEl = document.querySelector('label[for="maskInput"]');
            if (labelEl) {
                // keep the label (so it looks the same), but stop its default behavior
                labelEl.addEventListener("click", (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                    window.location.reload(); // simple reset
                }, { once: true }); // only needs to bind once after load

                // update label text
                labelEl.textContent = "Clear mask";

                // hide the actual file input so clicks won't open the picker
                maskInput.style.display = "none";
            }
        };
        img.onerror = () => (maskStatus.textContent = "Failed to load PNG");
        img.src = url;
    });
}


// ---------- helpers ----------
function lmkPx(lmk, W, H) {
    return { x: lmk.x * W, y: lmk.y * H };
}

function buildFaceGeometry() {
    const geometry = new THREE.BufferGeometry();
    const vertexCount = UV_COORDS.length;

    const pos = new Float32Array(vertexCount * 3);
    const positionAttr = new THREE.BufferAttribute(pos, 3);
    geometry.setAttribute("position", positionAttr);

    const uv = new Float32Array(vertexCount * 2);
    for (let i = 0; i < vertexCount; i++) {
        const [u, v] = UV_COORDS[i];
        uv[i * 2 + 0] = u;
        uv[i * 2 + 1] = v;
    }
    geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geometry.setIndex(TRIANGULATION);
    return { geometry, positionAttr };
}

/* -------------------------
   Demo 1: Click image detect — WebGL (seam-free)
--------------------------*/
const imageContainers = document.getElementsByClassName("detectOnClick");
Array.from(imageContainers).forEach((container) => {
    const img = container.querySelector("img");
    if (img) img.addEventListener("click", (evt) => handleClickWebGL(container, evt.target));
});

async function handleClickWebGL(container, imageEl) {
    if (!faceLandmarker) return; // need model to get landmarks

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await faceLandmarker.setOptions({ runningMode });
    }

    // clear previous overlays
    const allCanvas = container.getElementsByClassName("canvas");
    for (let i = allCanvas.length - 1; i >= 0; i--) allCanvas[i].remove();

    const result = faceLandmarker.detect(imageEl);
    const lmks = result?.faceLandmarks?.[0];
    if (!lmks) return;

    container.style.position = "relative";

    const overlay = document.createElement("canvas");
    overlay.className = "canvas";
    overlay.width = imageEl.naturalWidth;
    overlay.height = imageEl.naturalHeight;
    Object.assign(overlay.style, {
        position: "absolute",
        left: "0px",
        top: "0px",
        width: `${imageEl.width}px`,
        height: `${imageEl.height}px`,
        zIndex: "2",
        pointerEvents: "none",
    });
    container.appendChild(overlay);

    const renderer = new THREE.WebGLRenderer({ canvas: overlay, alpha: true, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(overlay.width, overlay.height, false);
    renderer.setClearColor(0x000000, 0);

    const W = overlay.width, H = overlay.height;
    // Y grows downward in pixel space
    const camera = new THREE.OrthographicCamera(0, W, 0, H, -1000, 1000);
    camera.position.z = 1; camera.lookAt(0, 0, 0);

    const scene = new THREE.Scene();
    const { geometry, positionAttr } = buildFaceGeometry();

    // 1) Wireframe UNDER mask
    const wire = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({
            color: 0xC0C0C0, wireframe: true, transparent: true, opacity: 0.7, depthTest: false
        })
    );
    wire.renderOrder = 0;
    wire.visible = showMesh;
    scene.add(wire);

    // 2) Mask ON TOP (only if a mask exists)
    if (maskImg) {
        const tex = new THREE.Texture(maskImg);
        tex.flipY = false; // correct vertical for image path
        // If your image-click demo needs horizontal flip too, uncomment next 3 lines:
        // tex.wrapS = THREE.RepeatWrapping; tex.repeat.x = -1; tex.offset.x = 1;
        tex.needsUpdate = true;

        const maskMesh = new THREE.Mesh(
            geometry,
            new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false })
        );
        maskMesh.renderOrder = 1;
        scene.add(maskMesh);
    }

    // Fill positions
    const pos = positionAttr.array;
    for (let i = 0; i < UV_COORDS.length; i++) {
        const p = lmkPx(lmks[i], W, H);
        const x = MIRROR_IMAGE ? (W - p.x) : p.x;
        const y = p.y;
        pos[i * 3 + 0] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = 0;
    }
    positionAttr.needsUpdate = true;

    renderer.render(scene, camera);
    drawBlendShapes(imageBlendShapes, result?.faceBlendshapes || []);
}

/* -------------------------
   Demo 2: Webcam detect — WebGL (seam-free)
--------------------------*/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas"); // WebGL canvas

let threeReady = false;
let renderer, scene, camera, mesh, wireMesh, positionAttr;

function initThreeWebcam(width, height) {
    if (threeReady) return;

    renderer = new THREE.WebGLRenderer({ canvas: canvasElement, alpha: true, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height, false);
    renderer.setClearColor(0x000000, 0);

    // Y grows downward like 2D canvas (same as image path)
    camera = new THREE.OrthographicCamera(0, width, 0, height, -1000, 1000);
    camera.position.z = 1; camera.lookAt(0, 0, 0);

    scene = new THREE.Scene();

    // shared texture (WEBCAM path)
    maskTexture = new THREE.Texture(maskImg || document.createElement("canvas"));
    maskTexture.flipY = false; // vertical orientation for webcam
    maskTexture.wrapS = THREE.RepeatWrapping;
    maskTexture.repeat.x = -1; // horizontal flip so text reads correctly
    maskTexture.offset.x = 1;
    maskTexture.needsUpdate = true;

    const built = buildFaceGeometry();
    const geometry = built.geometry;
    positionAttr = built.positionAttr;

    wireMesh = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({
            color: 0xC0C0C0, wireframe: true, transparent: true, opacity: 0.7, depthTest: false
        })
    );
    wireMesh.renderOrder = 0;
    wireMesh.visible = showMesh;
    scene.add(wireMesh);

    mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({
            map: maskTexture, transparent: true, depthTest: false, depthWrite: false
        })
    );
    mesh.renderOrder = 1;
    mesh.visible = !!maskImg; // only show if a mask already exists
    scene.add(mesh);

    threeReady = true;
}

function updateThreeSize(width, height) {
    if (!threeReady) return;
    renderer.setSize(width, height, false);
    camera.left = 0;
    camera.right = width;
    camera.top = 0;
    camera.bottom = height;
    camera.updateProjectionMatrix();
}

// webcam controls
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
    document.getElementById("webcamButton").innerText = webcamRunning ? "DISABLE PREDICTIONS" : "ENABLE WEBCAM";
    if (!webcamRunning) return;

    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let lastVideoTime = -1;
let results;

async function predictWebcam() {
    const ratio = video.videoHeight / Math.max(1, video.videoWidth);
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * ratio + "px";
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * ratio + "px";

    const W = video.videoWidth || canvasElement.width || 640;
    const H = video.videoHeight || canvasElement.height || 480;
    canvasElement.width = W;
    canvasElement.height = H;

    initThreeWebcam(W, H);
    updateThreeSize(W, H);

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode });
    }

    const startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }

    if (results?.faceLandmarks?.length) {
        const dstLmks = results.faceLandmarks[0];
        const pos = positionAttr.array;

        for (let i = 0; i < UV_COORDS.length; i++) {
            const p = lmkPx(dstLmks[i], W, H);
            const x = MIRROR_VIDEO ? (W - p.x) : p.x;
            const y = p.y;
            pos[i * 3 + 0] = x;
            pos[i * 3 + 1] = y;
            pos[i * 3 + 2] = 0;
        }
        positionAttr.needsUpdate = true;

        if (wireMesh) wireMesh.visible = showMesh;

        // hot-swap mask if uploaded during session
        if (maskImg && maskTexture.image !== maskImg) {
            maskTexture.image = maskImg;
            maskTexture.wrapS = THREE.RepeatWrapping;
            maskTexture.repeat.x = -1;
            maskTexture.offset.x = 1;
            maskTexture.needsUpdate = true;

            if (mesh && mesh.material) {
                mesh.material.map = maskTexture;
                mesh.material.transparent = true;
                mesh.material.depthTest = false;
                mesh.material.depthWrite = false;
                mesh.material.needsUpdate = true;
            }
        }

        // show mask iff a mask image exists
        if (mesh) mesh.visible = !!maskImg;
    }

    renderer.render(scene, camera);

    drawBlendShapes(videoBlendShapes, results?.faceBlendshapes || []);
    if (webcamRunning === true) window.requestAnimationFrame(predictWebcam);
}

// ---------- UI helpers (make sure this is NOT nested) ----------
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
