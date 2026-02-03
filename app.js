/* Videodroid — realtime visual synth (WebGL)
   FULL REPLACE — includes:
   - mobile-safe upload (no JS click() for file input)
   - reliable video load/play on iOS (tap canvas to start)
   - UNPACK_FLIP_Y_WEBGL for correct orientation
   - overlay (image/video) + opacity knob
   - CRT knob (kCrt) in shader
   - animate effects checkbox (freezes shader time when off)
   - export frame + webm + mp4 (via ffmpeg.wasm)
*/

const $ = (sel) => document.querySelector(sel);

const canvas = $("#glcanvas");
const dropzone = $("#dropzone");
const fileInput = $("#fileInput");

const videoEl = $("#srcVideo");
const imgEl = $("#srcImage");

const ovlVideoEl = $("#ovlVideo");
const ovlImgEl = $("#ovlImage");

const statusLeft = $("#statusLeft");
const statusRight = $("#statusRight");

const btnExportImg = $("#btnExportImg");
const btnRec = $("#btnRec");
const btnExportMp4 = $("#btnExportMp4");
const downloadLink = $("#downloadLink");

const btnFitContain = $("#btnFitContain");
const btnFitCover = $("#btnFitCover");
const btnClear = $("#btnClear");

const btnOverlay = $("#btnOverlay");
const btnClearOverlay = $("#btnClearOverlay");
const overlayInput = $("#overlayInput");

const tglMirror = $("#tglMirror");
const tglFreeze = $("#tglFreeze");
const tglAnimate = $("#tglAnimate");

/* -------------------- knob engine -------------------- */

function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
function formatVal(v){
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10)  return v.toFixed(2);
  return v.toFixed(3);
}

function setDialRotation(dialEl, inputEl){
  const min = parseFloat(inputEl.min);
  const max = parseFloat(inputEl.max);
  const val = parseFloat(inputEl.value);
  const t = (val - min) / (max - min);
  const deg = -135 + t * 270;
  dialEl.style.setProperty("--rot", `${deg}deg`);
}

function bindKnob(dialEl){
  const inputId = dialEl.dataset.for;
  const inputEl = document.getElementById(inputId);
  if (!inputEl) return;

  if (!inputEl.dataset.default) inputEl.dataset.default = inputEl.value;
  const valEl = document.getElementById("v" + inputId.slice(1)); // kHue -> vHue

  const update = () => {
    setDialRotation(dialEl, inputEl);
    if (valEl) valEl.textContent = formatVal(parseFloat(inputEl.value));
    inputEl.dispatchEvent(new Event("input", { bubbles:true }));
  };

  update();

  let dragging = false;
  let startY = 0;
  let startVal = 0;

  const min = parseFloat(inputEl.min);
  const max = parseFloat(inputEl.max);
  const step = parseFloat(inputEl.step || "0.001");

  function onDown(e){
    e.preventDefault();
    dragging = true;
    startY = (e.touches ? e.touches[0].clientY : e.clientY);
    startVal = parseFloat(inputEl.value);
    dialEl.setPointerCapture?.(e.pointerId);
  }

  function onMove(e){
    if (!dragging) return;
    const y = (e.touches ? e.touches[0].clientY : e.clientY);
    const dy = startY - y;

    const range = (max - min);
    const fine = e.shiftKey ? 0.25 : 1.0;
    const delta = (dy / 140) * range * 0.25 * fine;

    let next = startVal + delta;
    next = Math.round(next / step) * step;
    next = clamp(next, min, max);

    inputEl.value = String(next);
    update();
  }

  function onUp(){ dragging = false; }

  dialEl.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);

  dialEl.addEventListener("touchstart", onDown, { passive:false });
  window.addEventListener("touchmove", onMove, { passive:false });
  window.addEventListener("touchend", onUp);

  dialEl.addEventListener("dblclick", () => {
    inputEl.value = inputEl.dataset.default;
    update();
  });

  inputEl.addEventListener("input", () => {
    setDialRotation(dialEl, inputEl);
    if (valEl) valEl.textContent = formatVal(parseFloat(inputEl.value));
  });
}

function initKnobs(){
  document.querySelectorAll(".dial[data-for]").forEach(bindKnob);
  // fill any readouts not yet updated
  document.querySelectorAll("input[type=range].rangeHidden").forEach(inp=>{
    const vId = "v" + inp.id.slice(1);
    const vEl = document.getElementById(vId);
    if (vEl) vEl.textContent = formatVal(parseFloat(inp.value));
  });
}
initKnobs();

let fitMode = "contain";
btnFitContain.addEventListener("click", () => { fitMode = "contain"; });
btnFitCover.addEventListener("click", () => { fitMode = "cover"; });

/* -------------------- WebGL setup -------------------- */

const gl = canvas.getContext("webgl", { antialias:false, premultipliedAlpha:false, preserveDrawingBuffer:true });
if (!gl) {
  alert("WebGL not supported in this browser.");
  throw new Error("WebGL not supported");
}

const VERT = `
attribute vec2 aPos;
varying vec2 vUv;
void main(){
  vUv = (aPos + 1.0) * 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uSrc;
uniform sampler2D uFb;
uniform sampler2D uOvl;

uniform vec2 uRes;
uniform float uTime;

uniform float uHasSrc;
uniform float uMirror;
uniform float uFreeze;
uniform float uFitCover;
uniform vec2  uSrcSize;

uniform float uHasOvl;
uniform vec2  uOvlSize;
uniform float uOvlOpacity;

uniform float uHue;
uniform float uSat;
uniform float uCon;
uniform float uBri;

uniform float uFbAmt;
uniform float uFbRot;
uniform float uFbZoom;
uniform float uFbShift;

uniform float uWarp;
uniform float uScan;
uniform float uGlitch;
uniform float uSym;

uniform float uDistA;
uniform float uDistB;
uniform float uCrush;
uniform float uBleed;

uniform float uEdge;

uniform float uKey;
uniform float uSolar;
uniform float uPost;
uniform float uThresh;

uniform vec4  uChroma;
uniform float uChRate;

uniform vec4  uDirt;   // x grain, y speck, z vig, w chroma noise
uniform float uCrt;    // crt amount

float hash21(vec2 p){
  p = fract(p*vec2(123.34, 456.21));
  p += dot(p, p+45.32);
  return fract(p.x*p.y);
}
float hash31(vec3 p){
  p = fract(p*0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x+p.y)*p.z);
}

mat2 rot(float a){
  float s = sin(a), c = cos(a);
  return mat2(c,-s,s,c);
}

vec3 rgb2hsv(vec3 c){
  vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0*d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c){
  vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec2 fitUv(vec2 uv, vec2 srcSize, vec2 dstSize, float coverMode){
  float srcAR = srcSize.x / srcSize.y;
  float dstAR = dstSize.x / dstSize.y;
  vec2 outUv = uv;

  if (coverMode < 0.5) { // contain
    if (dstAR > srcAR) {
      float scale = srcAR / dstAR;
      outUv.x = (uv.x - 0.5) / scale + 0.5;
    } else {
      float scale = dstAR / srcAR;
      outUv.y = (uv.y - 0.5) / scale + 0.5;
    }
  } else { // cover
    if (dstAR > srcAR) {
      float scale = dstAR / srcAR;
      outUv.y = (uv.y - 0.5) * scale + 0.5;
    } else {
      float scale = srcAR / dstAR;
      outUv.x = (uv.x - 0.5) * scale + 0.5;
    }
  }
  return outUv;
}

vec3 postColor(vec3 col){
  vec3 h = rgb2hsv(col);
  h.x = fract(h.x + uHue);
  h.y *= uSat;
  col = hsv2rgb(h);

  col = (col - 0.5) * uCon + 0.5;
  col += uBri;

  if (uSolar > 0.001) {
    float m = smoothstep(0.0, 1.0, uSolar);
    vec3 inv = 1.0 - col;
    col = mix(col, mix(col, inv, step(0.5, col)), m);
  }

  if (uPost > 0.001) {
    float levels = mix(256.0, 3.0, clamp(uPost, 0.0, 1.0));
    col = floor(col * levels) / levels;
  }
  return clamp(col, 0.0, 1.0);
}

vec2 distortStage(vec2 uv, float amt, float seed){
  if (amt < 0.001) return uv;
  float t = uTime * (0.7 + 0.6*seed);

  uv.x += sin((uv.y*6.5 + t*1.3) * 6.2831853) * 0.010 * amt;
  uv.y += sin((uv.x*5.2 - t*1.1) * 6.2831853) * 0.012 * amt;

  vec2 n = vec2(
    hash21(uv*vec2(120.0, 80.0) + t),
    hash21(uv*vec2(70.0, 140.0) - t)
  ) - 0.5;
  uv += n * (0.006 * amt);

  float band = floor((uv.y + t*0.15) * 18.0);
  float rnd  = hash21(vec2(band, floor(t*7.0)));
  uv.x += (rnd - 0.5) * 0.08 * amt * smoothstep(0.35, 0.85, rnd);

  return uv;
}

float sobelEdgeLuma(sampler2D tex, vec2 uv){
  vec2 px = 1.0 / uRes;
  float tl = luma(texture2D(tex, uv + px*vec2(-1.0, -1.0)).rgb);
  float  l = luma(texture2D(tex, uv + px*vec2(-1.0,  0.0)).rgb);
  float bl = luma(texture2D(tex, uv + px*vec2(-1.0,  1.0)).rgb);
  float  t = luma(texture2D(tex, uv + px*vec2( 0.0, -1.0)).rgb);
  float  b = luma(texture2D(tex, uv + px*vec2( 0.0,  1.0)).rgb);
  float tr = luma(texture2D(tex, uv + px*vec2( 1.0, -1.0)).rgb);
  float  r = luma(texture2D(tex, uv + px*vec2( 1.0,  0.0)).rgb);
  float br = luma(texture2D(tex, uv + px*vec2( 1.0,  1.0)).rgb);

  float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
  float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
  return sqrt(gx*gx + gy*gy);
}

/* CRT shaping: subtle barrel + mask + vignette */
vec2 crtWarp(vec2 uv, float amt){
  vec2 p = uv*2.0 - 1.0;
  float r2 = dot(p,p);
  float k = mix(0.0, 0.18, amt);
  p *= 1.0 + k*r2;
  return p*0.5 + 0.5;
}

vec3 shadowMask(vec2 uv, float amt, vec3 col){
  if (amt < 0.001) return col;
  float x = uv.x * uRes.x;
  float m = sin(x*3.14159);
  float s = mix(1.0, 0.92 + 0.08*m, amt);
  col *= s;
  // tiny RGB mask
  float tri = fract(x/3.0);
  vec3 mask = vec3(
    smoothstep(0.0,0.6,1.0-abs(tri-0.2)),
    smoothstep(0.0,0.6,1.0-abs(tri-0.5)),
    smoothstep(0.0,0.6,1.0-abs(tri-0.8))
  );
  col *= mix(vec3(1.0), 0.92 + 0.08*mask, amt);
  return col;
}

void main(){
  vec2 uv = vUv;

  // CRT warp before anything else (so it feels like display)
  if (uCrt > 0.001) {
    uv = crtWarp(uv, uCrt);
  }

  if (uMirror > 0.5) uv.x = 1.0 - uv.x;

  if (uSym > 0.001) {
    float s = mix(1.0, 8.0, uSym);
    vec2 p = uv * 2.0 - 1.0;
    float a = atan(p.y, p.x);
    float r = length(p);
    float sector = 6.2831853 / s;
    a = mod(a + sector*0.5, sector) - sector*0.5;
    p = vec2(cos(a), sin(a)) * r;
    uv = (p * 0.5) + 0.5;
  }

  if (uWarp > 0.001) {
    float w = uWarp;
    float t = uTime * 0.75;
    uv.y += sin((uv.x*6.0 + t)*3.14159) * 0.015 * w;
    uv.x += sin((uv.y*5.0 - t)*3.14159) * 0.012 * w;
    uv += (vec2(hash21(uv+uTime), hash21(uv*1.7-uTime)) - 0.5) * 0.004 * w;
  }

  if (uGlitch > 0.001) {
    float g = uGlitch;
    float band = floor((uv.y + uTime*0.25) * 18.0);
    float rnd = hash21(vec2(band, floor(uTime*8.0)));
    float tear = (rnd - 0.5) * 0.12 * g;
    uv.x += tear * smoothstep(0.2, 0.8, rnd);
  }

  uv = distortStage(uv, uDistA, 0.27);
  uv = distortStage(uv, uDistB, 0.71);

  vec2 srcUv = fitUv(uv, uSrcSize, uRes, uFitCover);
  float inBounds =
    step(0.0, srcUv.x) * step(0.0, srcUv.y) *
    step(srcUv.x, 1.0) * step(srcUv.y, 1.0);

  vec2 ovlUv = fitUv(uv, uOvlSize, uRes, uFitCover);
  float ovlInBounds =
    step(0.0, ovlUv.x) * step(0.0, ovlUv.y) *
    step(ovlUv.x, 1.0) * step(ovlUv.y, 1.0);

  float ph = uTime * uChRate;
  vec2 phaseVec = vec2(cos(ph), sin(ph)) * (1.0 / min(uRes.x, uRes.y)) * 220.0;
  vec2 offR = phaseVec * uChroma.x * 0.0025;
  vec2 offG = phaseVec * uChroma.y * 0.0025;
  vec2 offB = phaseVec * uChroma.z * 0.0025;

  vec3 src = vec3(0.0);
  if (uHasSrc > 0.5 && inBounds > 0.5) {
    vec3 sR = texture2D(uSrc, clamp(srcUv + offR, 0.0, 1.0)).rgb;
    vec3 sG = texture2D(uSrc, clamp(srcUv + offG, 0.0, 1.0)).rgb;
    vec3 sB = texture2D(uSrc, clamp(srcUv + offB, 0.0, 1.0)).rgb;
    src = vec3(sR.r, sG.g, sB.b);
  }

  vec2 fbUv = uv;
  fbUv = (fbUv - 0.5);
  fbUv = rot(uFbRot) * fbUv;
  fbUv *= uFbZoom;
  fbUv += 0.5;
  fbUv += vec2(uFbShift, -uFbShift) * 0.75;

  vec3 fb = texture2D(uFb, fbUv).rgb;

  float baseMix = uFbAmt;
  if (uFreeze > 0.5) baseMix = max(baseMix, 0.92);

  vec3 col = mix(src, fb, baseMix);

  // overlay blend
  if (uHasOvl > 0.5 && uOvlOpacity > 0.001 && ovlInBounds > 0.5) {
    vec3 ovl = texture2D(uOvl, clamp(ovlUv, 0.0, 1.0)).rgb;
    col = mix(col, ovl, clamp(uOvlOpacity, 0.0, 1.0));
  }

  if (uBleed > 0.001) {
    vec2 px = 1.0 / uRes;
    vec3 smear = vec3(0.0);
    smear += texture2D(uFb, fbUv + vec2(-2.0, 0.0)*px).rgb;
    smear += texture2D(uFb, fbUv + vec2(-1.0, 0.0)*px).rgb;
    smear += texture2D(uFb, fbUv).rgb;
    smear /= 3.0;
    col = mix(col, smear, uBleed * 0.55);
  }

  if (uKey > 0.001) {
    float k = uKey;
    float y = luma(col);
    float edge = smoothstep(uThresh - 0.12*k, uThresh + 0.12*k, y);
    col = mix(col, vec3(edge), k*0.9);
  }

  if (uEdge > 0.001) {
    float e = sobelEdgeLuma(uFb, fbUv);
    e = smoothstep(0.12, 0.55, e);
    col = mix(col, col + vec3(e), uEdge * 0.9);
  }

  if (uScan > 0.001) {
    float s = uScan;
    float line = sin((vUv.y * uRes.y) * 3.14159);
    float mask = mix(1.0, 0.82 + 0.18*line, s);
    col *= mask;
    float tri = sin(vUv.x * uRes.x * 0.75);
    col *= mix(1.0, 0.98 + 0.02*tri, s);
  }

  if (uCrush > 0.001) {
    float c = uCrush;
    float levels = mix(256.0, 10.0, c);
    col = floor(col * levels) / levels;
    col.r = floor(col.r * (levels*0.85)) / (levels*0.85);
  }

  // dirt
  float grain = uDirt.x;
  float speck = uDirt.y;
  float vig   = uDirt.z;
  float cnoi  = uDirt.w;

  if (grain > 0.001 || speck > 0.001 || cnoi > 0.001 || vig > 0.001) {
    vec2 p = vUv * uRes / 2.0;

    float g = (hash21(p + uTime*13.1) - 0.5);
    col += g * 0.08 * grain;

    float nR = hash31(vec3(p*1.02, uTime*9.7)) - 0.5;
    float nG = hash31(vec3(p*0.98, uTime*8.9)) - 0.5;
    float nB = hash31(vec3(p*1.01, uTime*10.3)) - 0.5;
    col += vec3(nR, nG, nB) * 0.06 * cnoi;

    float sp = step(0.995 - speck*0.015, hash21(p*0.7 + uTime*2.0));
    float sd = (hash21(p*1.3 - uTime*3.0) > 0.5) ? 1.0 : -1.0;
    col += sd * sp * 0.35 * speck;

    vec2 q = vUv * 2.0 - 1.0;
    float v = smoothstep(1.2, 0.25, dot(q,q));
    col *= mix(1.0, v, vig);
  }

  // CRT mask after everything
  if (uCrt > 0.001) {
    col = shadowMask(vUv, uCrt, col);
    // extra vignette
    vec2 q = vUv*2.0 - 1.0;
    float vv = smoothstep(1.18, 0.22, dot(q,q));
    col *= mix(1.0, vv, uCrt*0.9);
  }

  col = postColor(col);
  gl_FragColor = vec4(col, 1.0);
}
`;

function compileShader(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const msg = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(msg);
  }
  return sh;
}
function createProgram(vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compileShader(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compileShader(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const msg = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(msg);
  }
  return p;
}

const prog = createProgram(VERT, FRAG);
gl.useProgram(prog);

const aPos = gl.getAttribLocation(prog, "aPos");
const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  -1,-1,  1,-1,  -1, 1,
  -1, 1,  1,-1,   1, 1
]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

const U = {
  uSrc: gl.getUniformLocation(prog, "uSrc"),
  uFb: gl.getUniformLocation(prog, "uFb"),
  uOvl: gl.getUniformLocation(prog, "uOvl"),

  uRes: gl.getUniformLocation(prog, "uRes"),
  uTime: gl.getUniformLocation(prog, "uTime"),

  uHasSrc: gl.getUniformLocation(prog, "uHasSrc"),
  uMirror: gl.getUniformLocation(prog, "uMirror"),
  uFreeze: gl.getUniformLocation(prog, "uFreeze"),
  uFitCover: gl.getUniformLocation(prog, "uFitCover"),
  uSrcSize: gl.getUniformLocation(prog, "uSrcSize"),

  uHasOvl: gl.getUniformLocation(prog, "uHasOvl"),
  uOvlSize: gl.getUniformLocation(prog, "uOvlSize"),
  uOvlOpacity: gl.getUniformLocation(prog, "uOvlOpacity"),

  uHue: gl.getUniformLocation(prog, "uHue"),
  uSat: gl.getUniformLocation(prog, "uSat"),
  uCon: gl.getUniformLocation(prog, "uCon"),
  uBri: gl.getUniformLocation(prog, "uBri"),

  uFbAmt: gl.getUniformLocation(prog, "uFbAmt"),
  uFbRot: gl.getUniformLocation(prog, "uFbRot"),
  uFbZoom: gl.getUniformLocation(prog, "uFbZoom"),
  uFbShift: gl.getUniformLocation(prog, "uFbShift"),

  uWarp: gl.getUniformLocation(prog, "uWarp"),
  uScan: gl.getUniformLocation(prog, "uScan"),
  uGlitch: gl.getUniformLocation(prog, "uGlitch"),
  uSym: gl.getUniformLocation(prog, "uSym"),

  uDistA: gl.getUniformLocation(prog, "uDistA"),
  uDistB: gl.getUniformLocation(prog, "uDistB"),
  uCrush: gl.getUniformLocation(prog, "uCrush"),
  uBleed: gl.getUniformLocation(prog, "uBleed"),

  uEdge: gl.getUniformLocation(prog, "uEdge"),

  uKey: gl.getUniformLocation(prog, "uKey"),
  uSolar: gl.getUniformLocation(prog, "uSolar"),
  uPost: gl.getUniformLocation(prog, "uPost"),
  uThresh: gl.getUniformLocation(prog, "uThresh"),

  uChroma: gl.getUniformLocation(prog, "uChroma"),
  uChRate: gl.getUniformLocation(prog, "uChRate"),

  uDirt: gl.getUniformLocation(prog, "uDirt"),
  uCrt: gl.getUniformLocation(prog, "uCrt"),
};

// textures: source + overlay + feedback pingpong
function makeTex(w, h) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  return t;
}
function makeFbo(tex) {
  const f = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, f);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  return f;
}

let fbW = 2, fbH = 2;
let fbTexA = makeTex(fbW, fbH);
let fbTexB = makeTex(fbW, fbH);
let fbFboA = makeFbo(fbTexA);
let fbFboB = makeFbo(fbTexB);
let fbFlip = false;

const srcTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, srcTex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 2, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE,
  new Uint8Array([0,0,0,255, 0,0,0,255, 0,0,0,255, 0,0,0,255])
);

const ovlTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, ovlTex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 2, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE,
  new Uint8Array([0,0,0,255, 0,0,0,255, 0,0,0,255, 0,0,0,255])
);

// bind samplers
gl.uniform1i(U.uSrc, 0);
gl.uniform1i(U.uFb, 1);
gl.uniform1i(U.uOvl, 2);

function resize() {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(2, Math.floor(rect.width * dpr));
  const h = Math.max(2, Math.floor(rect.height * dpr));

  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);

    fbW = w; fbH = h;
    fbTexA = makeTex(fbW, fbH);
    fbTexB = makeTex(fbW, fbH);
    fbFboA = makeFbo(fbTexA);
    fbFboB = makeFbo(fbTexB);

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbFboA);
    gl.clearColor(0,0,0,1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbFboB);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}
window.addEventListener("resize", resize);
resize();

/* -------------------- Media state + helpers -------------------- */

let hasSrc = false;
let srcSize = { w: 1, h: 1 };
let srcType = "none";
let srcObjectUrl = null;

let hasOvl = false;
let ovlSize = { w: 1, h: 1 };
let ovlType = "none";
let ovlObjectUrl = null;

function setStatus(left, right) {
  if (left != null) statusLeft.textContent = left;
  if (right != null) statusRight.textContent = right;
}

function showDropzone(show) {
  dropzone.classList.toggle("hidden", !show);
}

/* -------------------- overlay load/clear -------------------- */

function clearOverlay() {
  hasOvl = false;
  ovlType = "none";
  ovlSize = { w: 1, h: 1 };

  ovlVideoEl.pause();
  ovlVideoEl.removeAttribute("src");
  ovlVideoEl.load();
  ovlImgEl.removeAttribute("src");

  if (ovlObjectUrl) {
    try { URL.revokeObjectURL(ovlObjectUrl); } catch {}
    ovlObjectUrl = null;
  }
  setStatus(null, "overlay cleared");
}
btnClearOverlay.addEventListener("click", clearOverlay);

function loadOverlayFile(file) {
  if (!file) return;

  if (ovlObjectUrl) {
    try { URL.revokeObjectURL(ovlObjectUrl); } catch {}
    ovlObjectUrl = null;
  }

  const url = URL.createObjectURL(file);
  ovlObjectUrl = url;

  const isVideo = file.type.startsWith("video/");
  const isImage = file.type.startsWith("image/");

  if (!isVideo && !isImage) {
    setStatus(null, "overlay unsupported");
    return;
  }

  setStatus(null, "loading overlay…");

  if (isVideo) {
    ovlType = "video";
    hasOvl = false;

    ovlVideoEl.pause();
    ovlVideoEl.muted = true;
    ovlVideoEl.loop = true;
    ovlVideoEl.playsInline = true;
    ovlVideoEl.setAttribute("playsinline", "");
    ovlVideoEl.setAttribute("webkit-playsinline", "");
    ovlVideoEl.preload = "auto";

    const finalize = () => {
      const w = ovlVideoEl.videoWidth || 0;
      const h = ovlVideoEl.videoHeight || 0;
      if (w > 0 && h > 0) {
        ovlSize = { w, h };
        hasOvl = true;
        setStatus(null, `overlay live (${w}×${h})`);
      }
    };

    ovlVideoEl.onloadedmetadata = finalize;
    ovlVideoEl.onloadeddata = finalize;
    ovlVideoEl.oncanplay = () => {
      finalize();
      ovlVideoEl.play().catch(() => setStatus(null, "tap canvas to start overlay"));
    };

    ovlVideoEl.src = url;
    ovlVideoEl.load();
  } else {
    ovlType = "image";
    hasOvl = false;

    ovlImgEl.onload = () => {
      ovlSize = { w: ovlImgEl.naturalWidth || 1, h: ovlImgEl.naturalHeight || 1 };
      hasOvl = true;
      setStatus(null, `overlay live (${ovlSize.w}×${ovlSize.h})`);
    };
    ovlImgEl.onerror = () => {
      setStatus(null, "overlay image failed");
      hasOvl = false;
      ovlType = "none";
    };

    ovlImgEl.src = url;
  }
}

btnOverlay.addEventListener("click", () => overlayInput.click());
overlayInput.addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  loadOverlayFile(f);
  overlayInput.value = "";
});

/* -------------------- source load/clear -------------------- */

function clearSource() {
  hasSrc = false;
  srcType = "none";
  srcSize = { w: 1, h: 1 };

  videoEl.pause();
  videoEl.removeAttribute("src");
  videoEl.load();
  imgEl.removeAttribute("src");

  if (srcObjectUrl) {
    try { URL.revokeObjectURL(srcObjectUrl); } catch {}
    srcObjectUrl = null;
  }

  // also clear overlay to avoid stale blends
  clearOverlay();

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbFboA);
  gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbFboB);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  showDropzone(true);
  setStatus("no source loaded", "ready");
}
btnClear.addEventListener("click", clearSource);

function loadFile(file) {
  if (!file) return;

  if (srcObjectUrl) {
    try { URL.revokeObjectURL(srcObjectUrl); } catch {}
    srcObjectUrl = null;
  }

  const url = URL.createObjectURL(file);
  srcObjectUrl = url;

  const isVideo = file.type.startsWith("video/");
  const isImage = file.type.startsWith("image/");

  if (!isVideo && !isImage) {
    setStatus("unsupported file", "try png/jpg/mp4/webm");
    return;
  }

  showDropzone(false);
  setStatus(`${file.name}`, "loading…");

  if (isVideo) {
    srcType = "video";
    hasSrc = false;

    videoEl.pause();
    videoEl.muted = true;
    videoEl.loop = true;
    videoEl.playsInline = true;
    videoEl.setAttribute("playsinline", "");
    videoEl.setAttribute("webkit-playsinline", "");
    videoEl.preload = "auto";

    const finalize = () => {
      const w = videoEl.videoWidth || 0;
      const h = videoEl.videoHeight || 0;
      if (w > 0 && h > 0) {
        srcSize = { w, h };
        hasSrc = true;
        setStatus(`${file.name} (${w}×${h})`, "live");
      }
    };

    videoEl.onloadedmetadata = finalize;
    videoEl.onloadeddata = finalize;
    videoEl.oncanplay = () => {
      finalize();
      videoEl.play().catch(() => setStatus(null, "tap canvas to start video"));
    };
    videoEl.onerror = () => {
      setStatus(`${file.name}`, "video failed");
      hasSrc = false;
      srcType = "none";
    };

    videoEl.src = url;
    videoEl.load();
  } else {
    srcType = "image";
    hasSrc = false;

    imgEl.onload = () => {
      srcSize = { w: imgEl.naturalWidth || 1, h: imgEl.naturalHeight || 1 };
      hasSrc = true;
      setStatus(`${file.name} (${srcSize.w}×${srcSize.h})`, "live");
    };
    imgEl.onerror = () => {
      setStatus(`${file.name}`, "image failed");
      hasSrc = false;
      srcType = "none";
    };

    imgEl.src = url;
  }
}

/* drag & drop */
["dragenter","dragover"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
  });
});
["dragleave","drop"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
  });
});
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  loadFile(f);
});

/* click-to-upload handled by native file input overlay */
fileInput.addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  loadFile(f);
  fileInput.value = "";
});

/* iOS gesture: tap canvas to start videos */
canvas.addEventListener("pointerdown", () => {
  if (srcType === "video" && videoEl.src) videoEl.play().catch(()=>{});
  if (ovlType === "video" && ovlVideoEl.src) ovlVideoEl.play().catch(()=>{});
}, { passive:true });

/* -------------------- Upload DOM media into GL textures -------------------- */

function updateSourceTexture() {
  if (!hasSrc) return;

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, srcTex);

  try {
    if (srcType === "video") {
      if (videoEl.readyState >= 2) {
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoEl);
      }
    } else if (srcType === "image") {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgEl);
    }
  } catch (_) {}
}

function updateOverlayTexture() {
  if (!hasOvl) return;

  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, ovlTex);

  try {
    if (ovlType === "video") {
      if (ovlVideoEl.readyState >= 2) {
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, ovlVideoEl);
      }
    } else if (ovlType === "image") {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, ovlImgEl);
    }
  } catch (_) {}
}

/* -------------------- Render loop -------------------- */

const t0 = performance.now();
function uniFloat(loc, val) { gl.uniform1f(loc, val); }
function uniVec2(loc, x, y) { gl.uniform2f(loc, x, y); }
function uniVec4(loc, a,b,c,d) { gl.uniform4f(loc, a,b,c,d); }
const get = (id) => parseFloat(document.getElementById(id)?.value ?? "0");

function render() {
  resize();

  const rawTime = (performance.now() - t0) / 1000;
  const time = (tglAnimate?.checked ?? true) ? rawTime : 0.0;

  if (!tglFreeze.checked) {
    updateSourceTexture();
    updateOverlayTexture();
  }

  const w = canvas.width, h = canvas.height;

  const fbReadTex = fbFlip ? fbTexA : fbTexB;
  const fbWriteFbo = fbFlip ? fbFboB : fbFboA;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbWriteFbo);
  gl.viewport(0, 0, w, h);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, fbReadTex);

  uniVec2(U.uRes, w, h);
  uniFloat(U.uTime, time);

  uniFloat(U.uHasSrc, hasSrc ? 1 : 0);
  uniFloat(U.uMirror, tglMirror.checked ? 1 : 0);
  uniFloat(U.uFreeze, tglFreeze.checked ? 1 : 0);
  uniFloat(U.uFitCover, fitMode === "cover" ? 1 : 0);
  uniVec2(U.uSrcSize, srcSize.w, srcSize.h);

  uniFloat(U.uHasOvl, hasOvl ? 1 : 0);
  uniVec2(U.uOvlSize, ovlSize.w, ovlSize.h);
  uniFloat(U.uOvlOpacity, get("kOvl"));

  uniFloat(U.uHue, get("kHue"));
  uniFloat(U.uSat, get("kSat"));
  uniFloat(U.uCon, get("kCon"));
  uniFloat(U.uBri, get("kBri"));

  uniFloat(U.uFbAmt, get("kFb"));
  uniFloat(U.uFbRot, get("kFbRot"));
  uniFloat(U.uFbZoom, get("kFbZoom"));
  uniFloat(U.uFbShift, get("kFbShift"));

  uniFloat(U.uWarp, get("kWarp"));
  uniFloat(U.uScan, get("kScan"));
  uniFloat(U.uGlitch, get("kGlitch"));
  uniFloat(U.uSym, get("kSym"));

  uniFloat(U.uDistA, get("kDistA"));
  uniFloat(U.uDistB, get("kDistB"));
  uniFloat(U.uCrush, get("kCrush"));
  uniFloat(U.uBleed, get("kBleed"));

  uniFloat(U.uEdge, get("kEdge"));

  uniFloat(U.uKey, get("kKey"));
  uniFloat(U.uSolar, get("kSol"));
  uniFloat(U.uPost, get("kPost"));
  uniFloat(U.uThresh, get("kThresh"));

  uniVec4(U.uChroma, get("kChR"), get("kChG"), get("kChB"), 0.0);
  uniFloat(U.uChRate, get("kChRate"));

  uniVec4(U.uDirt, get("kGrain"), get("kSpeck"), get("kVig"), get("kCNoise"));
  uniFloat(U.uCrt, get("kCrt"));

  gl.drawArrays(gl.TRIANGLES, 0, 6);

  // present
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, w, h);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, fbFlip ? fbTexB : fbTexA);

  // display feedback texture as final
  uniFloat(U.uHasSrc, 0);
  uniFloat(U.uHasOvl, 0);
  uniFloat(U.uFbAmt, 1.0);
  uniFloat(U.uFreeze, 1.0);

  gl.drawArrays(gl.TRIANGLES, 0, 6);

  fbFlip = !fbFlip;
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

/* -------------------- Export frame -------------------- */

btnExportImg.addEventListener("click", () => {
  const png = canvas.toDataURL("image/png");
  downloadLink.href = png;
  downloadLink.download = `videodroid_frame_${Date.now()}.png`;
  downloadLink.click();
  setStatus(null, "exported frame");
});

/* -------------------- Export webm (toggle record) -------------------- */

let rec = null;
let recChunks = [];
let isRec = false;

btnRec.addEventListener("click", async () => {
  if (!isRec) {
    const stream = canvas.captureStream(60);
    const opts = { mimeType: "video/webm;codecs=vp9", videoBitsPerSecond: 8_000_000 };

    try { rec = new MediaRecorder(stream, opts); }
    catch (_) { rec = new MediaRecorder(stream); }

    recChunks = [];
    rec.ondataavailable = (e) => { if (e.data && e.data.size > 0) recChunks.push(e.data); };
    rec.onstop = () => {
      const blob = new Blob(recChunks, { type: rec.mimeType || "video/webm" });
      const url = URL.createObjectURL(blob);
      downloadLink.href = url;
      downloadLink.download = `videodroid_export_${Date.now()}.webm`;
      downloadLink.click();
      setStatus(null, "exported webm");
    };

    rec.start();
    isRec = true;
    btnRec.textContent = "stop webm";
    setStatus(null, "recording webm…");
  } else {
    isRec = false;
    btnRec.textContent = "export webm";
    if (rec && rec.state !== "inactive") rec.stop();
  }
});

/* -------------------- Export mp4 (records then converts) -------------------- */

let mp4Rec = null;
let mp4Chunks = [];
let isMp4Rec = false;

async function exportMp4FromWebmBlob(webmBlob) {
  // requires ffmpeg.js loaded in index.html
  const FF = window.FFmpeg;
  if (!FF) {
    setStatus(null, "ffmpeg not loaded");
    return;
  }

  const { createFFmpeg, fetchFile } = FF;
  const ffmpeg = createFFmpeg({ log: false });

  setStatus(null, "loading ffmpeg…");
  await ffmpeg.load();

  setStatus(null, "converting to mp4…");
  ffmpeg.FS("writeFile", "in.webm", await fetchFile(webmBlob));

  // simple transcode: H.264/AAC container
  await ffmpeg.run(
    "-i", "in.webm",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "veryfast",
    "-crf", "23",
    "-movflags", "+faststart",
    "out.mp4"
  );

  const data = ffmpeg.FS("readFile", "out.mp4");
  const mp4Blob = new Blob([data.buffer], { type: "video/mp4" });
  const url = URL.createObjectURL(mp4Blob);

  downloadLink.href = url;
  downloadLink.download = `videodroid_export_${Date.now()}.mp4`;
  downloadLink.click();
  setStatus(null, "exported mp4");
}

btnExportMp4.addEventListener("click", async () => {
  if (!isMp4Rec) {
    const stream = canvas.captureStream(60);
    // record webm first
    let recorder;
    try {
      recorder = new MediaRecorder(stream, { mimeType:"video/webm;codecs=vp9", videoBitsPerSecond: 8_000_000 });
    } catch {
      recorder = new MediaRecorder(stream);
    }
    mp4Rec = recorder;
    mp4Chunks = [];

    mp4Rec.ondataavailable = (e) => { if (e.data && e.data.size > 0) mp4Chunks.push(e.data); };
    mp4Rec.onstop = async () => {
      const blob = new Blob(mp4Chunks, { type: mp4Rec.mimeType || "video/webm" });
      await exportMp4FromWebmBlob(blob);
    };

    mp4Rec.start();
    isMp4Rec = true;
    btnExportMp4.textContent = "stop mp4";
    setStatus(null, "recording mp4…");
  } else {
    isMp4Rec = false;
    btnExportMp4.textContent = "export mp4";
    if (mp4Rec && mp4Rec.state !== "inactive") mp4Rec.stop();
  }
});

/* init */
showDropzone(true);
setStatus("no source loaded", "ready");
