import './style.css';

// ===== NEURAL NETWORK =====
let w1 = null;
let w2 = null;

async function loadWeights() {
    const res = await fetch('/weights/weights.json');
    const data = await res.json();
    w1 = data.w1;
    w2 = data.w2;
    console.log('Weights loaded — draw a digit!');
}

function relu(x) { return x > 0 ? x : 0; }

function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function softmax(v) {
    const max = Math.max(...v);
    const exps = v.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}

function predict(pixels) {
    if (!w1 || !w2) return null;
    const hidden = [];
    for (let i = 0; i < 128; i++) hidden[i] = relu(dot(pixels, w1[i]));
    const logits = [];
    for (let i = 0; i < 10; i++) logits[i] = dot(hidden, w2[i]);
    const probs = softmax(logits);
    return { digit: probs.indexOf(Math.max(...probs)), probs };
}

// ===== UI ELEMENTS (must be before clearCanvas!) =====
const digitEl = document.getElementById('prediction-digit');
const confEl = document.getElementById('prediction-confidence');
const barsEl = document.getElementById('bars');

// Build confidence bars
for (let i = 0; i < 10; i++) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `<span class="bar-d">${i}</span><div class="bar-track"><div class="bar-fill" id="b${i}"></div></div><span class="bar-p" id="p${i}">—</span>`;
    barsEl.appendChild(row);
}

function showResult(r) {
    if (!r) {
        digitEl.textContent = '?';
        confEl.textContent = 'draw something';
        for (let i = 0; i < 10; i++) {
            document.getElementById(`b${i}`).style.width = '0%';
            document.getElementById(`b${i}`).className = 'bar-fill';
            document.getElementById(`p${i}`).textContent = '—';
        }
        return;
    }
    digitEl.textContent = r.digit;
    confEl.textContent = `${(r.probs[r.digit] * 100).toFixed(1)}% confidence`;
    for (let i = 0; i < 10; i++) {
        const pct = r.probs[i] * 100;
        const bar = document.getElementById(`b${i}`);
        bar.style.width = `${pct}%`;
        bar.className = i === r.digit ? 'bar-fill top' : 'bar-fill';
        document.getElementById(`p${i}`).textContent = pct >= 1 ? `${pct.toFixed(0)}%` : '—';
    }
}

// ===== CANVAS DRAWING =====
const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

let isDrawing = false;

function clearCanvas() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Reset stroke style after fill
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    isDrawing = false;
    showResult(null);
}

clearCanvas();

// --- Mouse events (scale CSS coords to canvas buffer coords) ---
function canvasXY(e) {
    const sx = canvas.width / canvas.offsetWidth;
    const sy = canvas.height / canvas.offsetHeight;
    return [e.offsetX * sx, e.offsetY * sy];
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const [x, y] = canvasXY(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.1, y + 0.1);
    ctx.stroke();
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const [x, y] = canvasXY(e);
    ctx.lineTo(x, y);
    ctx.stroke();
});

window.addEventListener('mouseup', () => {
    if (isDrawing) {
        isDrawing = false;
        runPredict();
    }
});

// --- Touch events ---
function touchPos(e) {
    const rect = canvas.getBoundingClientRect();
    const t = e.touches[0];
    return {
        x: (t.clientX - rect.left) * (canvas.width / rect.width),
        y: (t.clientY - rect.top) * (canvas.height / rect.height),
    };
}

canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDrawing = true;
    const p = touchPos(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x + 0.1, p.y + 0.1);
    ctx.stroke();
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const p = touchPos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
}, { passive: false });

canvas.addEventListener('touchend', () => {
    if (isDrawing) {
        isDrawing = false;
        runPredict();
    }
});

document.getElementById('clear-btn').addEventListener('click', clearCanvas);

// ===== MNIST-STYLE PREPROCESSING =====
// MNIST digits are: (1) fit into a 20x20 box preserving aspect ratio,
// (2) placed in 28x28 centered by center of mass. We replicate this.

function getPixels() {
    const W = canvas.width, H = canvas.height;
    const src = ctx.getImageData(0, 0, W, H);

    // --- Step 1: Find bounding box of non-black pixels ---
    let minX = W, minY = H, maxX = 0, maxY = 0;
    let found = false;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (src.data[(y * W + x) * 4] > 20) { // threshold to ignore noise
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                found = true;
            }
        }
    }

    if (!found) return new Float64Array(784); // blank canvas

    // --- Step 2: Crop and scale into 20x20 box (preserving aspect ratio) ---
    // using an offscreen canvas with bilinear interpolation (much better than block averaging)
    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;
    const targetSize = 20; // MNIST fits digits into 20x20

    // Scale to fit in 20x20, preserve aspect ratio
    const scale = targetSize / Math.max(cropW, cropH);
    const scaledW = Math.max(1, Math.round(cropW * scale));
    const scaledH = Math.max(1, Math.round(cropH * scale));

    // Offscreen canvas for high-quality downscale
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = scaledW;
    tmpCanvas.height = scaledH;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.imageSmoothingEnabled = true;
    tmpCtx.imageSmoothingQuality = 'high';

    // Draw cropped region scaled down
    tmpCtx.drawImage(canvas, minX, minY, cropW, cropH, 0, 0, scaledW, scaledH);
    const scaledData = tmpCtx.getImageData(0, 0, scaledW, scaledH);

    // --- Step 3: Place in 28x28 centered by center of mass ---
    // First, compute center of mass of the scaled image
    let massX = 0, massY = 0, totalMass = 0;
    for (let y = 0; y < scaledH; y++) {
        for (let x = 0; x < scaledW; x++) {
            const val = scaledData.data[(y * scaledW + x) * 4];
            massX += x * val;
            massY += y * val;
            totalMass += val;
        }
    }

    if (totalMass === 0) return new Float64Array(784);

    const comX = massX / totalMass; // center of mass in scaled image
    const comY = massY / totalMass;

    // We want the center of mass to land at (14, 14) — the center of 28x28
    // So offset = 14 - com (in the 28x28 coordinate space)
    // First, place the scaled image centered geometrically, then shift by CoM offset
    const geoOffX = Math.round((28 - scaledW) / 2);
    const geoOffY = Math.round((28 - scaledH) / 2);

    // Center of mass offset from geometric center of the scaled image
    const comOffX = Math.round(14 - (geoOffX + comX));
    const comOffY = Math.round(14 - (geoOffY + comY));

    // Final placement position
    const placeX = geoOffX + comOffX;
    const placeY = geoOffY + comOffY;

    // --- Step 4: Render into 28x28 using another offscreen canvas ---
    const outCanvas = document.createElement('canvas');
    outCanvas.width = 28;
    outCanvas.height = 28;
    const outCtx = outCanvas.getContext('2d');
    outCtx.fillStyle = '#000';
    outCtx.fillRect(0, 0, 28, 28);
    outCtx.drawImage(tmpCanvas, placeX, placeY);

    // Read final 28x28 pixels
    const outData = outCtx.getImageData(0, 0, 28, 28);
    const px = new Float64Array(784);
    for (let i = 0; i < 784; i++) {
        px[i] = outData.data[i * 4] / 255;
    }
    return px;
}

function runPredict() {
    const result = predict(getPixels());
    showResult(result);
}

// ===== INIT =====
loadWeights().catch(console.error);
