/* ============================================================
   HARQ Visualizer — script.js
   ============================================================ */

"use strict";

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ============================================================
// SEEDED RANDOM NUMBER GENERATOR (Seed = 7)
// ============================================================
let currentSeed = 7;
function seedPRNG(seed) { currentSeed = seed; }
function randomFloat() {
  let t = currentSeed += 0x6D2B79F5;
  t = Math.imul(t ^ t >>> 15, t | 1);
  t ^= t + Math.imul(t ^ t >>> 7, t | 61);
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}
function gaussianRng() {
  let u, v;
  do { u = randomFloat(); } while (u === 0);
  v = randomFloat();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ============================================================
// 1. QUANTIZATION MATRICES
// ============================================================
const Q_MATRIX = [
  [16,11,10,16,24,40,51,61],
  [12,12,14,19,26,58,60,55],
  [14,13,16,24,40,57,69,56],
  [14,17,22,29,51,87,80,62],
  [18,22,37,56,68,109,103,77],
  [24,35,55,64,81,104,113,92],
  [49,64,78,87,103,121,120,101],
  [72,92,95,98,112,100,103,99]
];

const C_MATRIX = [
  [17,18,24,47,99,99,99,99],
  [18,21,26,66,99,99,99,99],
  [24,26,56,99,99,99,99,99],
  [47,66,99,99,99,99,99,99],
  [99,99,99,99,99,99,99,99],
  [99,99,99,99,99,99,99,99],
  [99,99,99,99,99,99,99,99],
  [99,99,99,99,99,99,99,99]
];

function renderQMatrices() {
  function fill(id, matrix) {
    const div = document.getElementById(id);
    if (!div) return;
    div.innerHTML = '';
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const v = matrix[r][c];
        const cls = v >= 80 ? 'hi' : v <= 20 ? 'lo' : '';
        const cell = document.createElement('div');
        cell.className = `matrix-cell ${cls}`;
        cell.textContent = v;
        div.appendChild(cell);
      }
    }
  }
  fill('qMatrixY', Q_MATRIX);
  fill('qMatrixC', C_MATRIX);
}

function renderQuantizedBlock() {
  const div = document.getElementById('quantizedBlock');
  if (!div || !State.firstQuantizedBlock) return;
  div.innerHTML = '';
  for (let i = 0; i < 64; i++) {
    const v = State.firstQuantizedBlock[i];
    const cls = v === 0 ? 'lo' : (Math.abs(v) > 20 ? 'hi' : '');
    const cell = document.createElement('div');
    cell.className = `matrix-cell ${cls}`;
    cell.textContent = v;
    div.appendChild(cell);
  }
  document.getElementById('quantizedCard').classList.remove('hidden');
}

// ============================================================
// 2. MATH HELPERS (DCT, IDCT, Binary Conversion)
// ============================================================
function dct1d(arr) {
  const N = 8; const out = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) sum += arr[n] * Math.cos(Math.PI * k * (2 * n + 1) / (2 * N));
    out[k] = sum * (k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N));
  }
  return out;
}

function idct1d(arr) {
  const N = 8; const out = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    let sum = arr[0] * Math.sqrt(1 / N);
    for (let k = 1; k < N; k++) sum += arr[k] * Math.sqrt(2 / N) * Math.cos(Math.PI * k * (2 * n + 1) / (2 * N));
    out[n] = sum;
  }
  return out;
}

function dct2d(block) {
  const tmp = new Float64Array(64), out = new Float64Array(64);
  for (let r = 0; r < 8; r++) {
    const row = block.subarray(r * 8, r * 8 + 8), d = dct1d(row);
    for (let c = 0; c < 8; c++) tmp[r * 8 + c] = d[c];
  }
  for (let c = 0; c < 8; c++) {
    const col = new Float64Array(8);
    for (let r = 0; r < 8; r++) col[r] = tmp[r * 8 + c];
    const d = dct1d(col);
    for (let r = 0; r < 8; r++) out[r * 8 + c] = d[r];
  }
  return out;
}

function idct2d(block) {
  const tmp = new Float64Array(64), out = new Float64Array(64);
  for (let c = 0; c < 8; c++) {
    const col = new Float64Array(8);
    for (let r = 0; r < 8; r++) col[r] = block[r * 8 + c];
    const d = idct1d(col);
    for (let r = 0; r < 8; r++) tmp[r * 8 + c] = d[r];
  }
  for (let r = 0; r < 8; r++) {
    const row = tmp.subarray(r * 8, r * 8 + 8), d = idct1d(row);
    for (let c = 0; c < 8; c++) out[r * 8 + c] = d[c];
  }
  return out;
}

function intToBits16(val) {
  val = Math.max(-32768, Math.min(32767, Math.round(val)));
  const u = (val + 65536) & 0xFFFF;
  const bits = new Uint8Array(16);
  for (let i = 15; i >= 0; i--) bits[15 - i] = (u >> i) & 1;
  return bits;
}

function bits16ToInt(bits) {
  let u = 0;
  for (let i = 0; i < 16; i++) u = (u << 1) | (bits[i] & 1);
  if (u >= 32768) u -= 65536;
  return u;
}

// ============================================================
// 3. CHANNEL FUNCTIONS & Math Operations
// ============================================================
const CRC8_POLY = 0x2F;

function crcEncode(u) {
  let reg = 0;
  for (let i = 0; i < u.length; i++) {
    const msb = (reg >> 7) & 1;
    reg = ((reg << 1) & 0xFF) | (u[i] & 1);
    if (msb) reg ^= CRC8_POLY;
  }
  for (let i = 0; i < 8; i++) {
    const msb = (reg >> 7) & 1;
    reg = (reg << 1) & 0xFF;
    if (msb) reg ^= CRC8_POLY;
  }
  const crcBits = new Uint8Array(8);
  for (let i = 7; i >= 0; i--) crcBits[7 - i] = (reg >> i) & 1;
  const out = new Uint8Array(u.length + 8);
  out.set(u); out.set(crcBits, u.length);
  return out;
}

function crcCheck(xHat) {
  const uHat = xHat.slice(0, xHat.length - 8);
  const crcHat = xHat.slice(xHat.length - 8);
  const recomp = crcEncode(uHat).slice(-8);
  let passed = true;
  for (let i = 0; i < 8; i++) if (crcHat[i] !== recomp[i]) { passed = false; break; }
  return { passed, uHat };
}

function bpskMod(bits) {
  const sym = new Float64Array(bits.length);
  for (let i = 0; i < bits.length; i++) sym[i] = 1.0 - 2.0 * bits[i]; 
  return sym;
}

function awgnChannel(symbols, ebn0_db, codeRate = 1.0) {
  const ebn0_lin = Math.pow(10, ebn0_db / 10);
  const sigma2   = 1.0 / (2.0 * codeRate * ebn0_lin);
  const sigma    = Math.sqrt(sigma2);
  const out      = new Float64Array(symbols.length);
  for (let i = 0; i < symbols.length; i++) out[i] = symbols[i] + gaussianRng() * sigma;
  return out;
}

function hardDecision(softSymbols) {
  const bits = new Uint8Array(softSymbols.length);
  for (let i = 0; i < softSymbols.length; i++) bits[i] = softSymbols[i] < 0 ? 1 : 0;
  return bits;
}

function circShift(arr, shift) {
  const n = arr.length, out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = arr[(i - shift + n) % n];
  return out;
}

function xorArrays(a, b) {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] ^ b[i];
  return out;
}

function circShiftFloat(arr, shift) {
  const n = arr.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = arr[(i - shift + n) % n];
  return out;
}

// ============================================================
// 4. JPEG ENCODING
// ============================================================
function encodeChannel(channelData, h, w, qMatrix, saveFirstBlock = false) {
  const bitstream = [];
  let firstBlockSaved = false;

  for (let bi = 0; bi < h / 8; bi++) {
    for (let bj = 0; bj < w / 8; bj++) {
      const block = new Float64Array(64);
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          block[r * 8 + c] = channelData[(bi * 8 + r) * w + (bj * 8 + c)] - 128.0;
        }
      }
      const dctBlock = dct2d(block);
      const quantized = new Float64Array(64);

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const qval = Math.round(dctBlock[r * 8 + c] / qMatrix[r][c]);
          quantized[r * 8 + c] = qval;
          bitstream.push(...intToBits16(qval));
        }
      }
      
      if (saveFirstBlock && !firstBlockSaved) {
        State.firstQuantizedBlock = Array.from(quantized);
        firstBlockSaved = true;
      }
    }
  }
  return new Uint8Array(bitstream);
}

function decodeChannel(bitstream, h, w, qMatrix) {
  const channelData = new Uint8Array(h * w);
  let bIdx = 0;
  for (let bi = 0; bi < h / 8; bi++) {
    for (let bj = 0; bj < w / 8; bj++) {
      const coeffs = new Float64Array(64);
      for (let k = 0; k < 64; k++) {
        coeffs[k] = bits16ToInt(bitstream.slice(bIdx, bIdx + 16));
        bIdx += 16;
      }
      const dequant = new Float64Array(64);
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) dequant[r * 8 + c] = coeffs[r * 8 + c] * qMatrix[r][c];
      }
      const spatial = idct2d(dequant);
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          let v = spatial[r * 8 + c] + 128.0;
          channelData[(bi * 8 + r) * w + (bj * 8 + c)] = Math.max(0, Math.min(255, Math.round(v)));
        }
      }
    }
  }
  return channelData;
}

function encodeColorImageToBits(yData, cbData, crData, h, w) {
  const yBits = encodeChannel(yData, h, w, Q_MATRIX, true);
  const cbBits = encodeChannel(cbData, h, w, C_MATRIX);
  const crBits = encodeChannel(crData, h, w, C_MATRIX);
  const all = new Uint8Array(yBits.length + cbBits.length + crBits.length);
  all.set(yBits, 0); all.set(cbBits, yBits.length); all.set(crBits, yBits.length + cbBits.length);
  return all;
}

function decodeBitsToColorImage(bitstream, h, w) {
  const bitsPerCh = (h / 8) * (w / 8) * 64 * 16;
  return {
    yData: decodeChannel(bitstream.slice(0, bitsPerCh), h, w, Q_MATRIX),
    cbData: decodeChannel(bitstream.slice(bitsPerCh, bitsPerCh * 2), h, w, C_MATRIX),
    crData: decodeChannel(bitstream.slice(bitsPerCh * 2, bitsPerCh * 3), h, w, C_MATRIX)
  };
}

function rgbToYCbCr(r, g, b) {
  return [
    Math.max(0, Math.min(255, Math.round(0.299 * r + 0.587 * g + 0.114 * b))),
    Math.max(0, Math.min(255, Math.round(-0.168736 * r - 0.331264 * g + 0.5 * b + 128))),
    Math.max(0, Math.min(255, Math.round(0.5 * r - 0.418688 * g - 0.081312 * b + 128)))
  ];
}
function yCbCrToRgb(y, cb, cr) {
  cb -= 128; cr -= 128;
  return [
    Math.max(0, Math.min(255, Math.round(y + 1.402 * cr))),
    Math.max(0, Math.min(255, Math.round(y - 0.344136 * cb - 0.714136 * cr))),
    Math.max(0, Math.min(255, Math.round(y + 1.772 * cb)))
  ];
}

// ============================================================
// 5. APPLICATION STATE
// ============================================================
const State = {
  imageLoaded : false,
  imgW: 0, imgH: 0,
  yData: null, cbData: null, crData: null, rgbaData: null,
  imageBits: null, jpegBits: null,
  firstQuantizedBlock: null,

  mode: 'A', ebn0: 5.0, maxM: 3, blockK: 1024, threshold: 4.0,

  simRunning: false, simPaused: false, fastForward: false,
  simBlocks: [], simCurrentBlock: 0,
  totalBlocks: 0, ackCount: 0, nackCount: 0, retxCount: 0, bitErrors: 0,
  receivedBits: null,

  animParticles: [], animRaf: null, blockDefs: [], activeBlkIdx: -1, activeBlkStatus: 'idle',
  pendingStep: false,
};

const $ = id => document.getElementById(id);
const dropZone = $('dropZone'); const imageInput = $('imageInput');
const encodingSection = $('encoding-section'); const harqSection = $('harq-section'); const resultsSection = $('results-section');

// ============================================================
// 6. IMAGE UPLOAD & MODAL UI
// ============================================================
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => { 
  e.preventDefault(); 
  dropZone.classList.remove('drag-over'); 
  handleFile(e.dataTransfer.files[0]); 
});

dropZone.addEventListener('click', (e) => { 
  if (e.target !== imageInput) imageInput.click(); 
});

imageInput.addEventListener('change', (e) => { 
  if (e.target.files[0]) {
    handleFile(e.target.files[0]);
    e.target.value = ''; 
  }
});

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = e => { const img = new Image(); img.onload = () => processImage(img); img.src = e.target.result; };
  reader.readAsDataURL(file);
}

function processImage(imgEl) {
  const w = imgEl.naturalWidth - (imgEl.naturalWidth % 8);
  const h = imgEl.naturalHeight - (imgEl.naturalHeight % 8);
  
  const maxDim = 384; 
  const scale = Math.min(1, maxDim / Math.max(w, h));
  const fw = Math.floor(w * scale / 8) * 8 || 8;
  const fh = Math.floor(h * scale / 8) * 8 || 8;

  State.imgW = fw; State.imgH = fh;
  const tmpCanvas = document.createElement('canvas'); tmpCanvas.width = fw; tmpCanvas.height = fh;
  const ctx = tmpCanvas.getContext('2d'); ctx.drawImage(imgEl, 0, 0, fw, fh);
  const rgba = ctx.getImageData(0, 0, fw, fh).data;
  
  const yArr = new Uint8Array(fw * fh), cbArr = new Uint8Array(fw * fh), crArr = new Uint8Array(fw * fh);
  State.rgbaData = new Uint8Array(rgba.buffer);

  for (let i = 0; i < fw * fh; i++) {
    const [y, cb, cr] = rgbToYCbCr(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]);
    yArr[i] = y; cbArr[i] = cb; crArr[i] = cr;
  }
  State.yData = yArr; State.cbData = cbArr; State.crData = crArr;

  const origCanvas = $('originalCanvas');
  origCanvas.width = fw; origCanvas.height = fh;
  origCanvas.getContext('2d').putImageData(ctx.getImageData(0, 0, fw, fh), 0, 0);
  makeClickable(origCanvas); 
  
  $('image-preview-row').classList.remove('hidden');
  State.imageLoaded = true;
  startJpegPipeline();
}

function makeClickable(canvas) {
  canvas.classList.add('clickable-canvas');
  canvas.onclick = () => {
    $('modalImage').src = canvas.toDataURL('image/jpeg', 1.0);
    $('imageModal').classList.remove('hidden');
  };
}

const imgModal = $('imageModal');
if (imgModal) {
  imgModal.onclick = (e) => {
    if (e.target.id !== 'modalImage') imgModal.classList.add('hidden');
  };
}

function renderChannel(canvas, data, w, h, name) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const id = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const val = data[i];
    const oi = i * 4;
    if (name === 'Y') {
      id.data[oi] = val; id.data[oi+1] = val; id.data[oi+2] = val;
    } else if (name === 'Cb') {
      id.data[oi] = 0; id.data[oi+1] = Math.round(val * 0.6); id.data[oi+2] = val;
    } else {
      id.data[oi] = val; id.data[oi+1] = 0; id.data[oi+2] = Math.round(val * 0.4);
    }
    id.data[oi+3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  makeClickable(canvas); 
}

const MODE_BLOCKS = {
  A: ['Data Block','CRC Encoder','BPSK Mod','AWGN Ch.','Chase Combiner','Hard Decision','CRC Check','ACK/NACK'],
  B: ['Data Block','CRC Encoder','Mother Code/RVs','Select RV','BPSK Mod','AWGN Ch.','Soft Buffer','IR Combiner','Hard Decision','CRC Check','ACK/NACK'],
  C: ['Data Block','Mother Code/RVs','Select RV','BPSK Mod','AWGN Ch.','Soft Buffer','Soft Combining','c1/c2 Candidates','Euclidean Dist.','Δ Threshold','ACK/NACK']
};

document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active'); 
    State.mode = btn.dataset.mode;
    $('modeDescription').querySelector('.card-label').textContent = btn.querySelector('.mode-name').textContent.replace('<br>',' ');
    $('thresholdRow').style.display = (State.mode === 'C') ? '' : 'none';
    
    const newM = (State.mode === 'A') ? 3 : 4;
    // تحديث الحد الأقصى للمتغير (M) في الـ Slider ليناسب النمط المختار
    $('slM').max = 10
    $('slM').value = newM;
    State.maxM = newM;
    $('valM').textContent = newM;

    renderBlockDiagram(); resetSimulation();
  });
});

const BLOCK_EXPLANATIONS = {
  'Data Block': 'One selected bit starts inside the data block.',
  'CRC Encoder': 'The bit passes through CRC encoding logic.',
  'BPSK Mod': 'The bit is converted into a BPSK wireless symbol.',
  'AWGN Ch.': 'The symbol travels through a noisy wireless channel.',
  'Chase Combiner': 'Received copies are combined to improve reliability.',
  'Hard Decision': 'The receiver decides whether the signal is a 0 or a 1.',
  'CRC Check': 'The receiver checks whether the block is valid.',
  'ACK/NACK': 'The receiver sends ACK for success or NACK for retransmission.',

  'Mother Code/RVs': 'The bit enters the mother code and redundancy version logic.',
  'Select RV': 'One redundancy version is selected for transmission.',
  'Soft Buffer': 'The received soft value is stored.',
  'IR Combiner': 'Incremental redundancy information is combined.',
  'Soft Combining': 'Soft wireless observations are combined.',
  'c1/c2 Candidates': 'Two candidate decoded bit sequences are compared.',
  'Euclidean Dist.': 'Distance is measured to estimate reliability.',
  'Δ Threshold': 'The confidence gap is compared with the threshold.'
};

function renderBlockDiagram() {
  const blocks = MODE_BLOCKS[State.mode] || [];
  const diagram = $('blockDiagram');

  diagram.innerHTML = '';
  diagram.classList.add('bit-sim-diagram');
  State.blockDefs = [];

  blocks.forEach((label, i) => {
    const blk = document.createElement('div');
    blk.className = 'blk';
    blk.id = `blk-${i}`;
    blk.textContent = label;

    diagram.appendChild(blk);
    State.blockDefs.push({ label, index: i });

    if (i < blocks.length - 1) {
      const cable = document.createElement('span');
      cable.className = 'blk-arrow bit-cable';
      cable.textContent = '────';
      diagram.appendChild(cable);
    }
  });

  $('blockExp').innerHTML = 'Press <strong>Start Simulation</strong> to watch one bit move through the full system.';
  State.activeBlkIdx = -1;
  State.activeBlkStatus = 'idle';
}

function setBlockActive(idx, status = 'active') {
  const blocks = MODE_BLOCKS[State.mode] || [];

  blocks.forEach((label, i) => {
    const el = $(`blk-${i}`);
    if (!el) return;

    if (i < idx) el.className = 'blk done-past';
    else if (i === idx) el.className = `blk ${status}`;
    else el.className = 'blk';
  });

  if (idx >= 0 && idx < blocks.length) {
    const label = blocks[idx];
    const explanation = BLOCK_EXPLANATIONS[label] || 'The bit is moving through this block.';
    $('blockExp').innerHTML = `<span class="exp-label">${label}:</span> ${explanation}`;
  }

  State.activeBlkIdx = idx;
  State.activeBlkStatus = status;
}

function pickDemoBitFromTx(txInfo) {
  const bits = State.paddedBits || State.imageBits;

  // For a NACK, prefer an actually corrupted data bit.
  // This keeps the animation honest: the displayed 0->1 or 1->0 flip is real.
  if (txInfo && txInfo.demoBit) {
    return txInfo.demoBit;
  }

  if (!bits || !bits.length) {
    const fallback = Math.random() < 0.5 ? 0 : 1;
    return {
      index: -1,
      txBit: fallback,
      rxBit: fallback,
      flipped: false,
      note: 'Demo bit only'
    };
  }

  if (typeof State.demoBitCursor !== 'number') {
    State.demoBitCursor = 0;
  }

  const blockStart = (State.simCurrentBlock || 0) * (State.blockK || 0);
  let bitIndex = blockStart + (State.demoBitCursor % (State.blockK || bits.length));

  if (bitIndex >= bits.length) {
    bitIndex = State.demoBitCursor % bits.length;
  }

  const bitValue = bits[bitIndex];

  State.demoBitCursor++;
  State.currentDemoBit = bitValue;

  return {
    index: bitIndex - blockStart,
    txBit: bitValue,
    rxBit: bitValue,
    flipped: false,
    note: 'No flip detected for this displayed bit'
  };
}

function makeDemoBitInfo(u, recoveredBits, passed, extraNote = '') {
  let preferredZeroError = -1;
  let firstError = -1;
  let errorCount = 0;

  for (let i = 0; i < u.length; i++) {
    if (recoveredBits[i] !== u[i]) {
      errorCount++;

      if (firstError < 0) {
        firstError = i;
      }

      // Prefer showing 0 -> 1 because this matches your explanation question.
      if (preferredZeroError < 0 && u[i] === 0) {
        preferredZeroError = i;
      }
    }
  }

  const chosen = preferredZeroError >= 0 ? preferredZeroError : firstError;

  if (!passed && chosen >= 0) {
    return {
      index: chosen,
      txBit: u[chosen],
      rxBit: recoveredBits[chosen],
      flipped: true,
      errorCount,
      note: extraNote || `Example corrupted data bit: data[${chosen}]`
    };
  }

  return {
    index: firstError,
    txBit: firstError >= 0 ? u[firstError] : (u[0] || 0),
    rxBit: firstError >= 0 ? recoveredBits[firstError] : (u[0] || 0),
    flipped: false,
    errorCount,
    note: passed
      ? 'ACK: no data-bit flip remains after combining/decision'
      : 'CRC failed, but the remaining error is in CRC/parity bits rather than the displayed data bit'
  };
}

function getNextDemoBitValue() {
  const bits = State.paddedBits || State.imageBits;

  if (!bits || !bits.length) {
    return Math.random() < 0.5 ? 0 : 1;
  }

  if (typeof State.demoBitCursor !== 'number') {
    State.demoBitCursor = 0;
  }

  const blockStart = (State.simCurrentBlock || 0) * (State.blockK || 0);
  let bitIndex = blockStart + (State.demoBitCursor % (State.blockK || bits.length));

  if (bitIndex >= bits.length) {
    bitIndex = State.demoBitCursor % bits.length;
  }

  const bitValue = bits[bitIndex];

  State.demoBitCursor++;
  State.currentDemoBit = bitValue;

  return bitValue;
}

function getDiagramPoint(el, parent, side = 'center') {
  const r = el.getBoundingClientRect();
  const p = parent.getBoundingClientRect();

  let x = r.left - p.left + parent.scrollLeft + r.width / 2;
  let y = r.top - p.top + parent.scrollTop + r.height / 2;

  if (side === 'right') x = r.right - p.left + parent.scrollLeft;
  if (side === 'left') x = r.left - p.left + parent.scrollLeft;

  return { x, y };
}

function placeMovingBit(bitEl, x, y) {
  bitEl.style.transform = `translate(${x - 13}px, ${y - 13}px)`;
}

function moveMovingBit(bitEl, from, to, duration) {
  return new Promise(resolve => {
    const startTime = performance.now();

    function frame(now) {
      const t = Math.min(1, (now - startTime) / duration);
      const smooth = t * t * (3 - 2 * t);

      const x = from.x + (to.x - from.x) * smooth;
      const y = from.y + (to.y - from.y) * smooth;

      placeMovingBit(bitEl, x, y);

      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    }

    requestAnimationFrame(frame);
  });
}

async function animateSingleBitThroughDiagram(txInfo, slow = false) {
  const blocks = MODE_BLOCKS[State.mode] || [];
  const diagram = $('blockDiagram');

  if (!blocks.length || !diagram) return;

  const finalPassed = !!(txInfo && txInfo.passed);
  const txNum = txInfo && txInfo.txNum ? txInfo.txNum : 1;
  const demoBit = pickDemoBitFromTx(txInfo);

  const bitValue = demoBit.txBit;
  let displayedBit = bitValue;
  let flipShown = false;

  const bitEl = document.createElement('div');
  bitEl.className = 'moving-bit';
  bitEl.textContent = bitValue;
  bitEl.title = `Transmit ${txNum}: data bit ${demoBit.index >= 0 ? demoBit.index : '?'} starts as ${bitValue}`;

  diagram.appendChild(bitEl);

  const moveTime = slow ? 650 : 280;
  const waitTime = slow ? 260 : 90;

  const firstBlock = $('blk-0');
  if (!firstBlock) {
    bitEl.remove();
    return;
  }

  setBlockActive(0, 'active');

  let currentPoint = getDiagramPoint(firstBlock, diagram, 'center');
  placeMovingBit(bitEl, currentPoint.x, currentPoint.y);

  $('blockExp').innerHTML +=
    ` <span class="current-bit-label">TX data bit${demoBit.index >= 0 ? ' #' + demoBit.index : ''}: <strong>${bitValue}</strong></span>`;

  await sleep(waitTime);

  for (let i = 1; i < blocks.length; i++) {
    const nextBlock = $(`blk-${i}`);
    if (!nextBlock) continue;

    const nextPoint = getDiagramPoint(nextBlock, diagram, 'center');

    bitEl.classList.add('in-wireless-link');
    await moveMovingBit(bitEl, currentPoint, nextPoint, moveTime);
    bitEl.classList.remove('in-wireless-link');

    const isLast = i === blocks.length - 1;
    const status = isLast ? (finalPassed ? 'success' : 'fail') : 'active';

    setBlockActive(i, status);

    // Educational visualization:
    // AWGN affects the analog BPSK symbol.
    // The actual 0/1 error appears after the noisy value is hard-decided.
    // We show the flip immediately at AWGN so the student can see why CRC gives NACK.
    if (!flipShown && demoBit.flipped && blocks[i] === 'AWGN Ch.') {
      displayedBit = demoBit.rxBit;
      bitEl.textContent = displayedBit;
      bitEl.classList.add('bit-flipped');

      bitEl.title =
        `Noise pushed this example bit across the decision boundary: ${demoBit.txBit} -> ${demoBit.rxBit}`;

      $('blockExp').innerHTML +=
        ` <span class="current-bit-label flip-note">Noise flip: <strong>${demoBit.txBit} -> ${demoBit.rxBit}</strong></span>`;

      flipShown = true;
    } else {
      $('blockExp').innerHTML +=
        ` <span class="current-bit-label">Displayed bit: <strong>${displayedBit}</strong></span>`;
    }

    currentPoint = nextPoint;

    await sleep(waitTime);
  }

  bitEl.classList.add(finalPassed ? 'bit-ack' : 'bit-nack');
  await sleep(slow ? 500 : 180);

  bitEl.remove();
}

// ============================================================
// 7. SIMULATION ENGINE
// ============================================================

$('btnStart').addEventListener('click', startSimulation);

$('btnPause').addEventListener('click', () => { 
  State.simPaused = !State.simPaused; 
  $('btnPause').textContent = State.simPaused ? '▶ Resume' : '⏸ Pause'; 
});

// تم نقل وظيفة البحث عن الأخطاء (NACK) إلى زر Step Next
let isSeekingNextNack = false;
$('btnStep').addEventListener('click', () => { 
  if (!State.simRunning) return;
  State.simPaused = false;
  isSeekingNextNack = true;
  $('btnPause').textContent = '⏸ Pause';
  $('btnPause').disabled = false;
  $('firstNackCard').classList.add('hidden');
});

// وظيفة زر التسريع السريع المباشر (True Fast Forward)
$('btnFastForward').addEventListener('click', () => { 
  State.fastForward = true; 
  State.simPaused = false; 
  $('btnFastForward').disabled = true; 
  $('btnPause').textContent = '⏸ Pause';
  $('btnPause').disabled = false;
});

$('btnReset').addEventListener('click', resetSimulation);

// ============================================================
// تم إصلاح خطأ استجابة الـ Sliders بالكامل هنا!
// ============================================================
$('slEbN0').addEventListener('input', e => { 
  State.ebn0 = parseFloat(e.target.value); 
  $('valEbN0').textContent = State.ebn0.toFixed(1) + ' dB'; 
});
$('slM').addEventListener('input', e => { 
  State.maxM = parseInt(e.target.value); 
  $('valM').textContent = State.maxM; 
});
$('slK').addEventListener('input', e => { 
  State.blockK = parseInt(e.target.value); 
  $('valK').textContent = State.blockK; 
});
$('slThreshold').addEventListener('input', e => { 
  State.threshold = parseFloat(e.target.value); 
  $('valThreshold').textContent = State.threshold.toFixed(1); 
});

function startJpegPipeline() {
  renderChannel($('yCanvas'), State.yData, State.imgW, State.imgH, 'Y');
  renderChannel($('cbCanvas'), State.cbData, State.imgW, State.imgH, 'Cb');
  renderChannel($('crCanvas'), State.crData, State.imgW, State.imgH, 'Cr');
  
  renderQMatrices();

  encodingSection.classList.remove('hidden');
  document.querySelectorAll('.pipe-step').forEach(el => el.className = 'pipe-step done');
  $('pipeExp').innerHTML = `<span class="exp-label">Success:</span> Image converted to bitstream successfully.`;

  setTimeout(() => {
    State.imageBits = encodeColorImageToBits(State.yData, State.cbData, State.crData, State.imgH, State.imgW);
    State.jpegBits = decodeBitsToColorImage(State.imageBits, State.imgH, State.imgW);
    
    renderQuantizedBlock();

    $('bitstreamDisplay').textContent = Array.from(State.imageBits.slice(0, 512)).join('');
    harqSection.classList.remove('hidden');
    $('btnStart').disabled = false; $('btnReset').disabled = false;
    renderBlockDiagram();
  }, 100);
}

function resetSimulation() {
  State.simRunning = false; State.simPaused = false; State.fastForward = false; isSeekingNextNack = false; State.firstNackShown = false;
  State.simCurrentBlock = 0; State.ackCount = 0; State.nackCount = 0; State.retxCount = 0; State.bitErrors = 0;
  $('btnStart').disabled = false; $('btnPause').disabled = true; $('btnStep').disabled = true; $('btnFastForward').disabled = true;
  $('simLog').innerHTML = '';
  $('firstNackCard').classList.add('hidden');
  const animCanvas = $('animCanvas');
  if (animCanvas) {
    const ctx = animCanvas.getContext('2d');
    ctx.clearRect(0, 0, animCanvas.width, animCanvas.height);
}
}

async function startSimulation() {
  resetSimulation();
  resultsSection.classList.remove('hidden');
  State.simRunning = true;
  $('btnStart').disabled = true; $('btnPause').disabled = false; $('btnStep').disabled = false; $('btnFastForward').disabled = false;
  
  seedPRNG(7); 
  State.demoBitCursor = 0;
  State.currentDemoBit = 0;

  const rawBits = State.imageBits;
  const k = State.blockK;
  const padLen = (-rawBits.length) % k;
  State.paddedBits = new Uint8Array(rawBits.length + (padLen>0?padLen:0));
  State.paddedBits.set(rawBits);
  State.totalBlocks = State.paddedBits.length / k;
  State.receivedBits = new Uint8Array(rawBits.length);

  //startAnimationLoop();
  await processBlocksLoop();
}

async function drawFirstNackGrid(origBits, rxBits, instant = false) {
  const canvas = $('firstNackCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 256, 256);
  
  const limit = Math.min(1024, origBits.length);
  const cellSize = 6;
  const spacing = 8;

  if (instant) {
    ctx.beginPath();
    for (let i = 0; i < limit; i++) {
      const isError = origBits[i] !== rxBits[i];
      ctx.fillStyle = isError ? '#ef4444' : '#22c55e';
      const x = (i % 32) * spacing + 1;
      const y = Math.floor(i / 32) * spacing + 1;
      ctx.fillRect(x, y, cellSize, cellSize);
    }
  } else {
    for (let row = 0; row < 32; row++) {
      for (let col = 0; col < 32; col++) {
        const i = row * 32 + col;
        if (i >= limit) break;
        const isError = origBits[i] !== rxBits[i];
        ctx.fillStyle = isError ? '#ef4444' : '#22c55e';
        ctx.fillRect(col * spacing + 1, row * spacing + 1, cellSize, cellSize);
      }
      await sleep(5); 
    }
  }
}

async function showNackDeepDive(idx, u, txHistory, instantGrid = false) {
  if (State.firstNackShown) return;

  State.firstNackShown = true;

  const card = $('firstNackCard');
  card.classList.remove('hidden');

  card.querySelector('.card-label').textContent = `Deep Dive: Analysis of Block ${idx + 1}`;
  $('firstNackLog').innerHTML = '';

  logLine(`<strong>--- ERROR DETECTED AT BLOCK ${idx + 1} ---</strong>`, 'log-warn', $('firstNackLog'));
  logLine(`<em>This block failed after ${State.maxM} transmission(s).</em>`, 'log-info', $('firstNackLog'));

  await drawFirstNackGrid(u, txHistory[0].recoveredBits, instantGrid);

  for (let tx of txHistory) {
    logLine(`<strong>Attempt ${tx.txNum}:</strong>`, '', $('firstNackLog'));

    tx.logs.forEach(msg => {
      logLine(`&nbsp;&nbsp;${msg}`, '', $('firstNackLog'));
    });

    if (!tx.passed && tx.errors > 0) {
      logLine(
        `&nbsp;&nbsp;<span class="log-flip">Flipped: ${tx.errors} bits corrupted by noise.</span>`,
        '',
        $('firstNackLog')
      );
    }
  }
}

async function processBlocksLoop() {
  while (State.simCurrentBlock < State.totalBlocks && State.simRunning) {
    
    while (State.simPaused && !State.fastForward && !isSeekingNextNack) {
      await sleep(100);
      if (!State.simRunning) return;
    }

    const idx = State.simCurrentBlock;
    const u = State.paddedBits.slice(idx * State.blockK, (idx + 1) * State.blockK);
    
    let txHistory;
    if (State.mode === 'A') txHistory = calcModeA(u, State.blockK, State.maxM, State.ebn0);
    else if (State.mode === 'B') txHistory = calcModeB(u, State.blockK, State.maxM, State.ebn0);
    else txHistory = calcModeC(u, State.blockK, State.maxM, State.ebn0, State.threshold);

    const isFirstBlock = (idx === 0); 
    const finalTx = txHistory[txHistory.length - 1];
    const isFinalNack = !finalTx.passed;
    
    // ---------------------------------------------------------
    // 1. Fast Forward Logic (No UI updates until the end)
    // ---------------------------------------------------------
    if (State.fastForward) {
  const rawLen = State.imageBits.length;
  const start = idx * State.blockK;

  for (let i = 0; i < State.blockK && (start + i) < rawLen; i++) {
    State.receivedBits[start + i] = finalTx.recoveredBits[i];

    if (State.imageBits[start + i] !== finalTx.recoveredBits[i]) {
      State.bitErrors++;
    }
  }

  if (finalTx.passed) {
    State.ackCount++;
  } else {
    State.nackCount++;

    // Show the first final NACK even during Fast Forward
    await showNackDeepDive(idx, u, txHistory, true);
  }

  State.retxCount += (finalTx.txNum - 1);
  State.simCurrentBlock++;

  if (idx % 50 === 0) {
    updateStats();
    drawBarChart();
    drawTimeline();
    await new Promise(r => setTimeout(r, 0));
  }

  continue;
}

    // ---------------------------------------------------------
    // 2. Step Next Logic (Search for NACK)
    // ---------------------------------------------------------
    if (isSeekingNextNack) {
      if (isFinalNack) {
        isSeekingNextNack = false;
        State.simPaused = true;
        $('btnPause').textContent = '▶ Resume';
        
        const card = $('firstNackCard');
        card.classList.remove('hidden');
        card.style.transition = 'box-shadow 0.2s';
        card.style.boxShadow = '0 0 30px rgba(249,115,22,0.6)';
        setTimeout(() => card.style.boxShadow = '0 0 20px rgba(239,68,68,0.2)', 300);

        card.querySelector('.card-label').textContent = `Deep Dive: Analysis of Block ${idx+1}`;
        $('firstNackLog').innerHTML = ''; 
        
        logLine(`<strong>--- ERROR DETECTED AT BLOCK ${idx+1} ---</strong>`, 'log-warn', $('firstNackLog'));
        logLine(`<em>Simulation paused. Found a block that failed after ${State.maxM} transmissions.</em>`, 'log-info', $('firstNackLog'));

        await drawFirstNackGrid(u, txHistory[0].recoveredBits, false); 

        for (let tx of txHistory) {
          logLine(`<strong>Attempt ${tx.txNum}:</strong>`, '', $('firstNackLog'));
          tx.logs.forEach(msg => logLine(`&nbsp;&nbsp;${msg}`, '', $('firstNackLog')));
          if (!tx.passed && tx.errors > 0) {
            logLine(`&nbsp;&nbsp;<span class="log-flip">Flipped: ${tx.errors} bits corrupted by noise.</span>`, '', $('firstNackLog'));
          }
        }
      } 
      else {
        const rawLen = State.imageBits.length;
        const start = idx * State.blockK;
        for (let i = 0; i < State.blockK && (start + i) < rawLen; i++) {
          State.receivedBits[start + i] = finalTx.recoveredBits[i];
          if (State.imageBits[start + i] !== finalTx.recoveredBits[i]) State.bitErrors++;
        }
        State.ackCount++;
        State.retxCount += (finalTx.txNum - 1);
        State.simCurrentBlock++;
        if (idx % 10 === 0) {
           updateStats(); drawBarChart(); drawTimeline();
           await new Promise(r => setTimeout(r, 0));
        }
        continue; 
      }
    }

    // ---------------------------------------------------------
    // 3. Normal Speed Logic
    // ---------------------------------------------------------
    // ---------------------------------------------------------
// 3. Normal Speed Logic
// ---------------------------------------------------------
if (!State.fastForward && !isSeekingNextNack) {
  if (isFirstBlock) {
    logLine(`<strong>--- BLOCK 1 TRACE ---</strong>`, 'log-info');
  }

  for (let tx of txHistory) {
    if (idx < 5 || isFinalNack) {
      logLine(`<strong>Block ${idx + 1} | Transmit ${tx.txNum}:</strong>`);
      tx.logs.forEach(msg => logLine(`&nbsp;&nbsp;${msg}`));
    }

    await animateSingleBitThroughDiagram(tx, isFirstBlock);
    await sleep(isFirstBlock ? 400 : 120);
  }

  if (isFinalNack) {
    const card = $('firstNackCard');
    card.classList.remove('hidden');

    card.querySelector('.card-label').textContent = `Deep Dive: Analysis of Block ${idx + 1}`;
    $('firstNackLog').innerHTML = '';

    logLine(`<strong>--- ERROR DETECTED AT BLOCK ${idx + 1} ---</strong>`, 'log-warn', $('firstNackLog'));
    logLine(`<em>This block failed after ${State.maxM} transmission(s).</em>`, 'log-info', $('firstNackLog'));

    await drawFirstNackGrid(u, txHistory[0].recoveredBits, false);

    for (let tx of txHistory) {
      logLine(`<strong>Attempt ${tx.txNum}:</strong>`, '', $('firstNackLog'));
      tx.logs.forEach(msg => logLine(`&nbsp;&nbsp;${msg}`, '', $('firstNackLog')));

      if (!tx.passed && tx.errors > 0) {
        logLine(
          `&nbsp;&nbsp;<span class="log-flip">Flipped: ${tx.errors} bits corrupted by noise.</span>`,
          '',
          $('firstNackLog')
        );
      }
    }
  }
}

    const rawLen = State.imageBits.length;
    const start = idx * State.blockK;
    for (let i = 0; i < State.blockK && (start + i) < rawLen; i++) {
      State.receivedBits[start + i] = finalTx.recoveredBits[i];
      if (State.imageBits[start + i] !== finalTx.recoveredBits[i]) State.bitErrors++;
    }

    if (finalTx.passed) State.ackCount++; else State.nackCount++;
    State.retxCount += (finalTx.txNum - 1);
    
    setBlockActive(MODE_BLOCKS[State.mode].length - 1, finalTx.passed ? 'success' : 'fail');
    
    State.simCurrentBlock++;

    updateStats(); drawBarChart(); drawTimeline();
    
    if (!isSeekingNextNack && !State.fastForward) {
        await sleep(30);
    }
  }

  updateStats(); drawBarChart(); drawTimeline();
  if (State.simRunning) finishSimulation();
}

// ============================================================
// 8. THE MATH ALGORITHMS 
// ============================================================
function calcModeA(u, k, M, ebn0) {
  const x = crcEncode(u);
  const codeRate = k / x.length;
  let rxBuffer = [];
  let txHistory = [];

  for (let tx = 1; tx <= M; tx++) {
    const txSym = bpskMod(x);
    const rxSym = awgnChannel(txSym, ebn0, codeRate);

    rxBuffer.push(rxSym);

    const combined = new Float64Array(rxSym.length);

    for (const buf of rxBuffer) {
      for (let i = 0; i < buf.length; i++) {
        combined[i] += buf[i];
      }
    }

    const xHat = hardDecision(combined);
    const { passed, uHat } = crcCheck(xHat);

    let errors = 0;
    for (let i = 0; i < u.length; i++) {
      if (uHat[i] !== u[i]) errors++;
    }

    const demoBit = makeDemoBitInfo(
      u,
      uHat,
      passed,
      'Example data bit selected from this NACK block'
    );

    const flipLog = demoBit.flipped
      ? `<span class="log-flip">Example flip: data[${demoBit.index}] ${demoBit.txBit} -> ${demoBit.rxBit} after AWGN + hard decision.</span>`
      : (!passed
          ? `<span class="log-flip">CRC failed, but the remaining visible data bits match; the error is likely in CRC bits.</span>`
          : null);

    txHistory.push({
      txNum: tx,
      passed,
      errors,
      recoveredBits: uHat,
      demoBit,
      logs: [
        `Sending exact copy.`,
        `Chase Combining applied across ${tx} buffer(s).`,
        flipLog,
        passed
          ? `<span class="log-ack">CRC PASS -> ACK</span>`
          : `<span class="log-nack">CRC FAIL -> NACK</span>`
      ].filter(Boolean)
    });

    if (passed) break;
  }

  return txHistory;
}

function calcModeB(u, k, M, ebn0) {
  const uCrc = crcEncode(u);

  const rvs = [
    uCrc,
    xorArrays(uCrc, circShift(uCrc, 1)),
    xorArrays(uCrc, circShift(uCrc, 3)),
    xorArrays(uCrc, circShift(uCrc, 5))
  ];

  let softBuf = [];
  let txHistory = [];

  for (let tx = 0; tx < M; tx++) {
    const rvIndex = tx % rvs.length;
    const rxSym = awgnChannel(bpskMod(rvs[rvIndex]), ebn0, 1.0);

    softBuf.push(rxSym);

    const combined = new Float64Array(uCrc.length);

    for (let i = 0; i < uCrc.length; i++) {
      combined[i] = softBuf[0][i];
    }

    if (tx >= 1 && softBuf[1]) {
      const sh1 = circShiftFloat(softBuf[0], 1);
      for (let i = 0; i < uCrc.length; i++) {
        combined[i] += softBuf[1][i] * sh1[i];
      }
    }

    if (tx >= 2 && softBuf[2]) {
      const sh3 = circShiftFloat(softBuf[0], 3);
      for (let i = 0; i < uCrc.length; i++) {
        combined[i] += softBuf[2][i] * sh3[i];
      }
    }

    if (tx >= 3 && softBuf[3]) {
      const sh5 = circShiftFloat(softBuf[0], 5);
      for (let i = 0; i < uCrc.length; i++) {
        combined[i] += softBuf[3][i] * sh5[i];
      }
    }

    const xHat = hardDecision(combined);
    const { passed, uHat } = crcCheck(xHat);

    let errors = 0;
    for (let i = 0; i < u.length; i++) {
      if (uHat[i] !== u[i]) errors++;
    }

    const demoBit = makeDemoBitInfo(
      u,
      uHat,
      passed,
      'Example data bit selected after IR combining'
    );

    const flipLog = demoBit.flipped
      ? `<span class="log-flip">Example flip: data[${demoBit.index}] ${demoBit.txBit} -> ${demoBit.rxBit} after AWGN + hard decision.</span>`
      : (!passed
          ? `<span class="log-flip">CRC failed, but the displayed data bit was not one of the remaining data errors.</span>`
          : null);

    txHistory.push({
      txNum: tx + 1,
      passed,
      errors,
      recoveredBits: uHat,
      demoBit,
      logs: [
        `Sending RV${rvIndex} (Parity slice).`,
        `Soft Combining ${tx + 1} RVs using multiplication.`,
        flipLog,
        passed
          ? `<span class="log-ack">CRC PASS -> ACK</span>`
          : `<span class="log-nack">CRC FAIL -> NACK</span>`
      ].filter(Boolean)
    });

    if (passed) break;
  }

  return txHistory;
}

function calcModeC(u, k, M, ebn0, threshold) {
  const rvs = [
    u,
    xorArrays(u, circShift(u, 1)),
    xorArrays(u, circShift(u, 3)),
    xorArrays(u, circShift(u, 5))
  ];

  let softBuf = [];
  let txHistory = [];

  for (let tx = 0; tx < Math.min(M, 4); tx++) {
    const rxSym = awgnChannel(bpskMod(rvs[tx]), ebn0, 1.0);

    softBuf.push(rxSym);

    const combined = new Float64Array(u.length);

    for (let i = 0; i < u.length; i++) {
      combined[i] = softBuf[0][i];
    }

    if (tx >= 1) {
      const sh1 = circShiftFloat(softBuf[0], 1);
      for (let i = 0; i < u.length; i++) {
        combined[i] += softBuf[1][i] * sh1[i];
      }
    }

    if (tx >= 2) {
      const sh3 = circShiftFloat(softBuf[0], 3);
      for (let i = 0; i < u.length; i++) {
        combined[i] += softBuf[2][i] * sh3[i];
      }
    }

    if (tx >= 3) {
      const sh5 = circShiftFloat(softBuf[0], 5);
      for (let i = 0; i < u.length; i++) {
        combined[i] += softBuf[3][i] * sh5[i];
      }
    }

    const c1Bits = hardDecision(combined);

    let weakIdx = 0;
    let weakAbs = Infinity;

    for (let i = 0; i < u.length; i++) {
      const a = Math.abs(combined[i]);
      if (a < weakAbs) {
        weakAbs = a;
        weakIdx = i;
      }
    }

    const c2Bits = new Uint8Array(c1Bits);
    c2Bits[weakIdx] ^= 1;

    const c1Rvs = [
      c1Bits,
      xorArrays(c1Bits, circShift(c1Bits, 1)),
      xorArrays(c1Bits, circShift(c1Bits, 3)),
      xorArrays(c1Bits, circShift(c1Bits, 5))
    ];

    const c2Rvs = [
      c2Bits,
      xorArrays(c2Bits, circShift(c2Bits, 1)),
      xorArrays(c2Bits, circShift(c2Bits, 3)),
      xorArrays(c2Bits, circShift(c2Bits, 5))
    ];

    let dC1 = 0;
    let dC2 = 0;

    for (let t = 0; t <= tx; t++) {
      const y = softBuf[t];
      const s1 = bpskMod(c1Rvs[t]);
      const s2 = bpskMod(c2Rvs[t]);

      for (let i = 0; i < u.length; i++) {
        dC1 += (y[i] - s1[i]) ** 2;
        dC2 += (y[i] - s2[i]) ** 2;
      }
    }

    const delta = dC2 - dC1;
    const passed = delta >= threshold;

    let errors = 0;
    for (let i = 0; i < u.length; i++) {
      if (c1Bits[i] !== u[i]) errors++;
    }

    const demoBit = makeDemoBitInfo(
      u,
      c1Bits,
      passed,
      'Example data bit selected after CRC-less soft decision'
    );

    const flipLog = demoBit.flipped
      ? `<span class="log-flip">Example flip: data[${demoBit.index}] ${demoBit.txBit} -> ${demoBit.rxBit} after AWGN + soft decision.</span>`
      : (!passed
          ? `<span class="log-flip">NACK came from low confidence, not necessarily from this displayed bit.</span>`
          : null);

    txHistory.push({
      txNum: tx + 1,
      passed,
      errors,
      recoveredBits: c1Bits,
      demoBit,
      logs: [
        `Euclidean Distance c1: ${dC1.toFixed(2)} | c2: ${dC2.toFixed(2)}`,
        `Delta Gap = ${delta.toFixed(2)} (Threshold = ${threshold.toFixed(2)})`,
        flipLog,
        passed
          ? `<span class="log-ack">Delta >= Threshold -> ACK</span>`
          : `<span class="log-nack">Delta < Threshold -> NACK</span>`
      ].filter(Boolean)
    });

    if (passed) break;
  }

  return txHistory;
}

// ============================================================
// 9. UI, LOGGING, & ANIMATION
// ============================================================
function finishSimulation() {
  State.simRunning = false;
  $('btnStart').disabled = false; $('btnPause').disabled = true; $('btnStep').disabled = true; $('btnFastForward').disabled = true;
  
  const ber = State.bitErrors / State.imageBits.length;
  logLine(`<br><span class="log-info">═══ Simulation Complete ═══</span>`);
  const animStatus = $('animStatus');
  if (animStatus) {
    animStatus.innerHTML = `<span style="color:var(--green)">✓ Complete.</span> BER: ${ber.toExponential(3)}`;
}
  updateStats(); renderReconstructedImages();
}

function updateStats() {
  $('st-total').textContent = State.totalBlocks;
  $('st-ack').textContent   = State.ackCount;
  $('st-nack').textContent  = State.nackCount;
  $('st-retx').textContent  = State.retxCount;
  $('st-errors').textContent= State.bitErrors;
  if (State.imageBits) $('st-ber').textContent = (State.bitErrors / State.imageBits.length).toExponential(2);
}

function logLine(html, type = 'info', target = $('simLog')) {
  const span = document.createElement('div');
  span.className = type;
  span.innerHTML = html;
  target.appendChild(span);
  target.scrollTop = target.scrollHeight;
}

function startAnimationLoop() {
  cancelAnimationFrame(State.animRaf);
  State.animParticles = [];
  function loop() { drawAnimCanvas(); State.animRaf = requestAnimationFrame(loop); }
  loop();
}

function spawnBlockParticles(passed, isSlow) {
  const w = $('animCanvas').width;
  const color = passed ? '#22c55e' : '#ef4444';
  const amount = isSlow ? 12 : 6;
  for (let i = 0; i < amount; i++) {
    State.animParticles.push({
      x: 10 + Math.random() * 20,
      y: 40 + Math.random() * 80,
      vx: (isSlow ? 2 : 5) + Math.random() * 2, 
      vy: (Math.random() - 0.5) * 1.5,
      radius: 5 + Math.random() * 3,
      color,
      label: Math.random() < 0.5 ? '0' : '1',
      life: 1.0,
      decay: isSlow ? 0.005 : 0.012
    });
  }
}

function drawAnimCanvas() {
  const canvas = $('animCanvas'), ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#0b0e14'; ctx.fillRect(0, 0, W, H);

  const blocks = MODE_BLOCKS[State.mode] || [];
  const blockW = Math.min(90, (W - 30) / blocks.length - 8);
  const gap    = (W - 20 - blocks.length * blockW) / (blocks.length - 1 || 1);
  const bH = 44, bY = H / 2 - bH / 2;

  blocks.forEach((label, i) => {
    const bX = 10 + i * (blockW + gap);
    const isActive = i === State.activeBlkIdx;
    
    ctx.beginPath(); ctx.roundRect(bX, bY, blockW, bH, 5);
    if (isActive) { ctx.fillStyle = 'rgba(59,130,246,0.18)'; ctx.strokeStyle = '#3b82f6'; }
    else { ctx.fillStyle = '#141a26'; ctx.strokeStyle = '#222d42'; }
    ctx.lineWidth = isActive ? 2 : 1; ctx.fill(); ctx.stroke();

    ctx.fillStyle = isActive ? '#93c5fd' : '#4b5563';
    ctx.font = `${Math.min(11, blockW / 7)}px 'Courier New'`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(label.substring(0, 12), bX + blockW / 2, bY + bH / 2);
    
    if (i < blocks.length - 1) { 
      ctx.strokeStyle = '#222d42'; ctx.beginPath(); ctx.moveTo(bX + blockW, bY + bH/2); ctx.lineTo(bX + blockW + gap, bY + bH/2); ctx.stroke();
    }
  });

  for (let i = State.animParticles.length - 1; i >= 0; i--) {
    const p = State.animParticles[i]; p.x += p.vx; p.y += p.vy; p.life -= p.decay;
    if (p.life <= 0 || p.x > W) { State.animParticles.splice(i, 1); continue; }
    
    ctx.globalAlpha = Math.min(1, p.life);
    ctx.beginPath(); ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
    ctx.fillStyle = p.color; ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = '9px monospace'; ctx.fillText(p.label, p.x, p.y);
  }
  ctx.globalAlpha = 1;
}

function drawTimeline() {
  const ctx = $('timelineCanvas').getContext('2d'), W = $('timelineCanvas').width, H = 80;
  ctx.fillStyle = '#0b0e14'; ctx.fillRect(0, 0, W, H);
  if (!State.totalBlocks) return;
  const cellW = W / State.totalBlocks;
  
  let drawn = 0;
  for(let i=0; i<State.ackCount; i++) { ctx.fillStyle='#22c55e'; ctx.fillRect(drawn*cellW, 20, cellW+0.5, 40); drawn++; }
  for(let i=0; i<State.nackCount; i++){ ctx.fillStyle='#ef4444'; ctx.fillRect(drawn*cellW, 20, cellW+0.5, 40); drawn++; }
}

function drawBarChart() {
  const ctx = $('barChart').getContext('2d'), W = $('barChart').width, H = 180;
  ctx.fillStyle = '#0b0e14'; ctx.fillRect(0, 0, W, H);
  const labels = ['Total Blocks', 'ACK Blocks', 'NACK Blocks', 'Retransmissions'];
  const vals = [State.totalBlocks, State.ackCount, State.nackCount, State.retxCount];
  const colors = ['#3b82f6', '#22c55e', '#ef4444', '#facc15'];
  const maxVal = Math.max(1, ...vals), barW = 80, gap = (W - 4 * barW) / 5;

  labels.forEach((lbl, i) => {
    const bH = (vals[i] / maxVal) * (H - 60), bX = gap + i * (barW + gap), bY = H - 40 - bH;
    ctx.fillStyle = colors[i]; ctx.fillRect(bX, bY, barW, bH);
    ctx.fillStyle = '#e2e8f0'; ctx.font = '13px monospace'; ctx.fillText(vals[i], bX + barW/2, bY - 6);
  });
}

function renderReconstructedImages() {
  const { imgW, imgH } = State;
  
  const origCanvas = $('reconOrigCanvas');
  origCanvas.width = imgW; origCanvas.height = imgH;
  origCanvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(State.rgbaData), imgW, imgH), 0, 0);
  makeClickable(origCanvas); 

  const { yData: jy, cbData: jcb, crData: jcr } = State.jpegBits;
  const jpegCanvas = $('reconJpegCanvas');
  drawYCbCr(jpegCanvas, jy, jcb, jcr, imgW, imgH);
  makeClickable(jpegCanvas); 

  if (State.receivedBits) {
    const { yData: ry, cbData: rcb, crData: rcr } = decodeBitsToColorImage(State.receivedBits, imgH, imgW);
    const rxCanvas = $('reconRxCanvas');
    drawYCbCr(rxCanvas, ry, rcb, rcr, imgW, imgH);
    makeClickable(rxCanvas); 
  }
}

function drawYCbCr(canvas, yData, cbData, crData, w, h) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d'), id = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const [r, g, b] = yCbCrToRgb(yData[i], cbData[i], crData[i]);
    id.data[i * 4] = r; id.data[i * 4 + 1] = g; id.data[i * 4 + 2] = b; id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}