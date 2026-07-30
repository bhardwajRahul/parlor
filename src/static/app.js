const $ = id => document.getElementById(id);
const video = $('video'), cameraToggle = $('cameraToggle');
const messagesDiv = $('messages'), statusEl = $('status');
const stateDot = $('stateDot'), stateText = $('stateText');
const viewportWrap = $('viewportWrap');
const waveformCanvas = $('waveform');
const waveformCtx = waveformCanvas.getContext('2d');

let ws, mediaStream, myvad;
let cameraEnabled = true;
let audioCtx;
let state = 'loading';
let ignoreIncomingAudio = false;

// Streaming audio playback state
let streamSampleRate = 24000;
let streamNextTime = 0;         // When to schedule next chunk
let streamSources = [];         // Active AudioBufferSourceNodes

// Per-state [glow, glow-dim]. Resolved values, not nested var(): setState
// writes them into CSS custom properties and the waveform paints with them.
const STATE_COLORS = {
  listening: ['#4ade80', 'rgba(74,222,128,0.12)'],
  processing: ['#f59e0b', 'rgba(245,158,11,0.12)'],
  speaking: ['#818cf8', 'rgba(129,140,248,0.12)'],
  loading: ['#3a3d46', 'rgba(58,61,70,0.12)'],
};

// ── Waveform Visualizer ──
let analyser, micSource;
const BAR_COUNT = 40;
const BAR_GAP = 3;
let waveformRAF;
let ambientPhase = 0;

function initWaveformCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = waveformCanvas.getBoundingClientRect();
  waveformCanvas.width = rect.width * dpr;
  waveformCanvas.height = rect.height * dpr;
  waveformCtx.scale(dpr, dpr);
}

function drawWaveform() {
  const { width: w, height: h } = waveformCanvas.getBoundingClientRect();
  waveformCtx.clearRect(0, 0, w, h);

  const barWidth = (w - (BAR_COUNT - 1) * BAR_GAP) / BAR_COUNT;
  waveformCtx.fillStyle = (STATE_COLORS[state] || STATE_COLORS.loading)[0];

  let dataArray = null;
  if (analyser) {
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
  }

  for (let i = 0; i < BAR_COUNT; i++) {
    let amplitude;
    if (dataArray) {
      // Map bar index to frequency bin
      const binIndex = Math.floor((i / BAR_COUNT) * dataArray.length * 0.6);
      amplitude = dataArray[binIndex] / 255;
    }

    // If no real audio data or very quiet, use ambient drift
    if (!dataArray || amplitude < 0.02) {
      ambientPhase += 0.0001;
      const drift = Math.sin(ambientPhase * 3 + i * 0.4) * 0.5 + 0.5;
      amplitude = 0.03 + drift * 0.04;
    }

    const barH = Math.max(2, amplitude * (h - 4));
    const x = i * (barWidth + BAR_GAP);
    const y = (h - barH) / 2;

    waveformCtx.globalAlpha = 0.3 + amplitude * 0.7;
    waveformCtx.beginPath();
    const r = Math.min(barWidth / 2, barH / 2, 3);
    waveformCtx.roundRect(x, y, barWidth, barH, r);
    waveformCtx.fill();
  }

  waveformCtx.globalAlpha = 1;
  waveformRAF = requestAnimationFrame(drawWaveform);
}

// ── Dynamic glow intensity for speaking state ──
function updateSpeakingGlow() {
  if (state !== 'speaking' || !analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  const avg = sum / data.length / 255;
  // Modulate the glow layers based on audio amplitude
  const intensity = 0.3 + avg * 0.7;
  viewportWrap.style.setProperty('--speak-intensity', intensity);
  const spread = 20 + avg * 60;
  const inner = 15 + avg * 25;
  viewportWrap.querySelector('.viewport-glow').style.boxShadow =
    `0 0 ${spread}px ${spread * 0.4}px rgba(129,140,248,${intensity * 0.25})`;
  viewportWrap.style.boxShadow =
    `inset 0 0 ${inner}px rgba(129,140,248,${intensity * 0.15}), 0 0 ${inner}px rgba(129,140,248,${intensity * 0.2})`;
  requestAnimationFrame(updateSpeakingGlow);
}

// ── State Machine ──
let processingWatchdog = null;

function setState(newState) {
  state = newState;

  stateDot.className = `dot ${newState}`;
  const labels = { loading: 'Loading...', listening: 'Listening', processing: 'Thinking...', speaking: 'Speaking' };
  stateText.textContent = labels[newState] || newState;

  viewportWrap.className = `viewport-wrap ${newState}`;

  // Reset inline styles from speaking glow
  if (newState !== 'speaking') {
    viewportWrap.style.boxShadow = '';
    viewportWrap.querySelector('.viewport-glow').style.boxShadow = '';
  }

  const [glow, glowDim] = STATE_COLORS[newState] || STATE_COLORS.loading;
  document.documentElement.style.setProperty('--glow', glow);
  document.documentElement.style.setProperty('--glow-dim', glowDim);

  if (newState === 'speaking') requestAnimationFrame(updateSpeakingGlow);

  // Raise VAD threshold during speaking to reduce echo false triggers
  if (myvad) {
    myvad.setOptions({ positiveSpeechThreshold: newState === 'speaking' ? 0.92 : 0.5 });
  }

  if (newState === 'listening' && mediaStream && audioCtx && analyser) {
    if (!micSource) {
      micSource = audioCtx.createMediaStreamSource(mediaStream);
    }
    try { micSource.connect(analyser); } catch {}
  } else if (micSource && newState !== 'listening') {
    try { micSource.disconnect(analyser); } catch {}
  }

  // Watchdog: never let 'processing' become a terminal state (e.g. the
  // server errored without sending a terminal frame).
  clearTimeout(processingWatchdog);
  if (newState === 'processing') {
    processingWatchdog = setTimeout(() => {
      if (state === 'processing') {
        console.warn('processing watchdog fired — returning to listening');
        goListening();
      }
    }, 30000);
  }
}

function goListening() { setState('listening'); setStatus('connected', 'Connected'); }
function goProcessing() { setState('processing'); setStatus('processing', 'Processing'); }

// ── WebSocket ──
function wsOpen() { return ws && ws.readyState === WebSocket.OPEN; }

function wsSend(payload) {
  if (wsOpen()) ws.send(JSON.stringify(payload));
}

function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => {
    setStatus('connected', 'Connected');
    if (state !== 'loading') setState('listening');
  };
  ws.onclose = () => {
    // A reconnect gets a brand-new server conversation — reset per-turn state.
    framePrefetched = false;
    clearFlushTimer();
    setStatus('disconnected', 'Disconnected');
    setTimeout(connect, 2000);
  };
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === 'text_delta') {
      if (ignoreIncomingAudio) return;
      appendAssistantText(msg.text);
    } else if (msg.type === 'turn_incomplete') {
      // The classifier judged the user unfinished — keep listening; if they
      // stay silent, flush the held audio so the model answers anyway.
      if (ignoreIncomingAudio) { ignoreIncomingAudio = false; return; }
      removeLastUserLoading();
      serverHoldingAudio = true;
      goListening();
      startFlushTimer();
    } else if (msg.type === 'transcript') {
      // The transcript line leads the model's reply, so what you said
      // appears while the response is still decoding.
      if (ignoreIncomingAudio) return;
      fillUserBubble(msg.transcription);
    } else if (msg.type === 'turn_final') {
      if (ignoreIncomingAudio) return;
      fillUserBubble(msg.transcription);
      const t = msg.timings || {};
      if (t.llm_time) {
        setAssistantMeta(`LLM ${t.llm_time}s` + (t.ttfa_s ? ` · audio @ ${t.ttfa_s}s` : ''));
      }
      if (!msg.spoke) goListening();  // no audio will follow
    } else if (msg.type === 'audio_start') {
      if (ignoreIncomingAudio) return;
      streamSampleRate = msg.sample_rate || 24000;
      startStreamPlayback();
    } else if (msg.type === 'audio_chunk') {
      if (ignoreIncomingAudio) return;
      queueAudioChunk(msg.audio);
    } else if (msg.type === 'audio_end') {
      if (ignoreIncomingAudio) {
        ignoreIncomingAudio = false;
        stopPlayback();
        setState('listening');
        return;
      }
      const meta = messagesDiv.querySelector('.msg.assistant:last-child .meta');
      if (meta) meta.textContent += ` · TTS ${msg.tts_time}s`;
    }
  };
}

// ── Streamed assistant message helpers ──
const LOADING_DOTS = '<span class="loading-dots"><span></span><span></span><span></span></span>';
let currentAssistantEl = null;
let currentAssistantBodyEl = null;

function appendAssistantText(text) {
  if (!currentAssistantEl) {
    currentAssistantEl = document.createElement('div');
    currentAssistantEl.className = 'msg assistant';
    currentAssistantEl.innerHTML = '<span class="body"></span>';
    currentAssistantBodyEl = currentAssistantEl.querySelector('.body');
    messagesDiv.appendChild(currentAssistantEl);
  }
  currentAssistantBodyEl.textContent += text;
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function setAssistantMeta(text) {
  if (!currentAssistantEl) return;
  let meta = currentAssistantEl.querySelector('.meta');
  if (!meta) {
    meta = document.createElement('div');
    meta.className = 'meta';
    currentAssistantEl.appendChild(meta);
  }
  meta.textContent = text;
}

function removeLastUserLoading() {
  const userMsgs = messagesDiv.querySelectorAll('.msg.user');
  const last = userMsgs[userMsgs.length - 1];
  if (last && last.querySelector('.loading-dots')) last.remove();
}

// Replace the last user bubble's loading dots with the heard words (no-op if
// already filled — the early 'transcript' frame wins over turn_final's copy).
function fillUserBubble(transcription) {
  const userMsgs = messagesDiv.querySelectorAll('.msg.user');
  const last = userMsgs[userMsgs.length - 1];
  if (last && last.querySelector('.loading-dots')) {
    const meta = last.querySelector('.meta');
    last.innerHTML = `${transcription || '…'}${meta ? meta.outerHTML : ''}`;
  }
}

// ── Flush timer: an incomplete turn went quiet — have the model answer the
// held audio anyway (it asks the user to continue if the words really are
// unfinished). Self-limiting: one flush per hold. serverHoldingAudio tracks
// the held state so a cleared timer (VAD misfire, barge-in) can re-arm —
// otherwise a cough after a false hold strands the question in silence.
let flushTimer = null;
let serverHoldingAudio = false;

function startFlushTimer() {
  clearFlushTimer();
  flushTimer = setTimeout(() => {
    flushTimer = null;
    // Never flush over live speech — that utterance will consume the held
    // audio itself when it ends.
    if (state === 'listening' && !speechActive && wsOpen()) {
      serverHoldingAudio = false;
      currentAssistantEl = null;
      ignoreIncomingAudio = false;
      addMessage('user', LOADING_DOTS);
      wsSend({ type: 'flush' });
      goProcessing();
    }
  }, 2500);
}

function clearFlushTimer() {
  if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
}

function setStatus(cls, text) { statusEl.className = `status-pill ${cls}`; statusEl.textContent = text; }

// ── Camera ──
const VIDEO_CONSTRAINTS = { width: 640, height: 480, facingMode: 'user' };
const AUDIO_CONSTRAINTS = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: VIDEO_CONSTRAINTS, audio: AUDIO_CONSTRAINTS,
    });
    video.srcObject = mediaStream;
    return;
  } catch (e) { console.warn('Video+audio failed:', e.message); }

  const streams = await Promise.allSettled([
    navigator.mediaDevices.getUserMedia({ video: VIDEO_CONSTRAINTS }),
    navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS }),
  ]);
  mediaStream = new MediaStream();
  streams.forEach(r => { if (r.status === 'fulfilled') r.value.getTracks().forEach(t => mediaStream.addTrack(t)); });
  if (mediaStream.getVideoTracks().length) video.srcObject = mediaStream;
  if (!mediaStream.getAudioTracks().length) { cameraEnabled = false; }
}

function captureFrame() {
  if (!cameraEnabled || !video.videoWidth) return null;
  const canvas = document.createElement('canvas');
  const scale = 320 / video.videoWidth;
  canvas.width = 320; canvas.height = video.videoHeight * scale;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
}

// ── VAD Handlers ──
let speakingStartedAt = 0;
let framePrefetched = false; // server holds an unconsumed camera frame
const BARGE_IN_GRACE_MS = 800; // Ignore VAD triggers shortly after TTS starts (echo)

// ── Speech chunk streaming ──
// While the user talks, ship ~3s audio chunks so the server can prefill
// them into the LLM cache; at speech end only the tail remains to process.
const CHUNK_SECONDS = 3.0;
const PRE_FRAMES = 10; // ~320ms of pre-speech context
let uttFrames = [];    // frames of the current utterance
let preFrames = [];    // rolling pre-speech buffer
let uttSamplesSent = 0;
let chunkSeq = 0;
let speechActive = false;

let bargeFrames = 0; // consecutive high-probability speech frames while TTS plays

function handleVadFrame(probs, frame) {
  // Barge-in needs ~250ms of SUSTAINED speech, not a single loud frame —
  // otherwise the mic catching our own TTS interrupts the reply (echo).
  if (state === 'speaking') {
    bargeFrames = (probs && probs.isSpeech > 0.92) ? bargeFrames + 1 : 0;
    if (bargeFrames >= 8) { bargeFrames = 0; triggerBargeIn(); }
    return;
  }
  bargeFrames = 0;
  if (speechActive && state === 'listening') {
    uttFrames.push(frame);
    const total = totalSamples(uttFrames);
    if (total - uttSamplesSent >= CHUNK_SECONDS * 16000) sendSpeechChunk(total);
  } else if (!speechActive) {
    preFrames.push(frame);
    if (preFrames.length > PRE_FRAMES) preFrames.shift();
  }
}

function totalSamples(frames) {
  let total = 0;
  for (const f of frames) total += f.length;
  return total;
}

function concatFrames(frames) {
  const out = new Float32Array(totalSamples(frames));
  let off = 0;
  for (const f of frames) { out.set(f, off); off += f.length; }
  return out;
}

function sendSpeechChunk(upTo) {
  if (!wsOpen()) return;
  const chunk = concatFrames(uttFrames).subarray(uttSamplesSent, upTo);
  if (chunk.length < 4800) return; // <300ms — not worth a priming request
  wsSend({ type: 'speech_chunk', seq: chunkSeq++, audio: float32ToWavBase64(chunk) });
  uttSamplesSent = upTo;
}

function startUtteranceCapture() {
  speechActive = true;
  uttFrames = [...preFrames];
  uttSamplesSent = 0;
  chunkSeq = 0;
}

// Send the camera frame the moment speech STARTS: the server prefills the
// image tokens while the user is still talking, so the audio turn only pays
// for the audio (~75% faster first token on camera turns).
function prefetchFrame() {
  if (framePrefetched || !cameraEnabled || !wsOpen()) return;
  const image = captureFrame();
  if (!image) return;
  wsSend({ type: 'frame', image });
  framePrefetched = true;
}

function handleSpeechStart() {
  clearFlushTimer();
  if (state === 'listening') { prefetchFrame(); startUtteranceCapture(); }
  // Barge-in while speaking is handled frame-by-frame in handleVadFrame,
  // which requires sustained speech — a single VAD trigger is too often echo.
}

function triggerBargeIn() {
  if (Date.now() - speakingStartedAt < BARGE_IN_GRACE_MS) {
    console.log('Barge-in suppressed (echo grace period)');
    return;
  }
  stopPlayback();
  ignoreIncomingAudio = true;
  wsSend({ type: 'interrupt' });
  setState('listening');
  prefetchFrame();
  startUtteranceCapture();
  console.log('Barge-in: interrupted playback');
}

function handleSpeechEnd(audio) {
  const wasStreaming = speechActive && uttSamplesSent > 0;
  speechActive = false; // always reset — a leak here streams mic audio forever
  if (state !== 'listening' || !wsOpen()) return;

  clearFlushTimer();
  serverHoldingAudio = false;
  // Frame already prefetched at speech start; capture now only as fallback.
  const imageBase64 = framePrefetched ? null : captureFrame();
  const withCamera = framePrefetched || !!imageBase64;
  framePrefetched = false;

  goProcessing();
  addMessage('user', LOADING_DOTS, withCamera ? 'with camera' : '');

  currentAssistantEl = null;
  ignoreIncomingAudio = false;
  const payload = {};
  if (wasStreaming) {
    // Chunks already streamed — send only the un-sent tail (may be empty;
    // the server then responds from the streamed chunks alone).
    const tail = concatFrames(uttFrames).subarray(uttSamplesSent);
    if (tail.length >= 1600) payload.audio = float32ToWavBase64(tail);
    payload.chunked = true;
  } else {
    payload.audio = float32ToWavBase64(audio);
  }
  if (imageBase64) payload.image = imageBase64;
  wsSend(payload);
}

// ── Float32 @ 16kHz → WAV base64 ──
function float32ToWavBase64(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0,'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8,'WAVE'); w(12,'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); w(36,'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = ''; for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

// ── Streaming Playback ──
function stopPlayback() {
  for (const src of streamSources) {
    try { src.stop(); } catch {}
  }
  streamSources = [];
  streamNextTime = 0;
}

let ttsSink = null;    // MediaStreamDestination the TTS plays into
let ttsOut = null;     // where TTS sources connect (sink, or raw destination)

// '?raw-audio' bypasses the AEC media-element route: cleaner, consistent
// output, but the mic will hear the assistant — headphones only. Useful to
// A/B macOS voice-processing artifacts the AEC path can trigger.
const RAW_AUDIO = new URLSearchParams(location.search).has('raw-audio');
let ttsAudioEl = null; // hidden <audio> element that actually outputs it

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.75;
    // Route TTS through a media element instead of audioCtx.destination:
    // Chrome's echo canceller only uses element/WebRTC playback as its
    // reference signal, so raw WebAudio output would be picked up by the
    // mic as "user speech" despite echoCancellation: true (crbug 687574).
    ttsSink = audioCtx.createMediaStreamDestination();
    ttsOut = RAW_AUDIO ? audioCtx.destination : ttsSink;
    ttsAudioEl = document.createElement('audio');
    ttsAudioEl.srcObject = ttsSink.stream;
    ttsAudioEl.autoplay = true;
    document.body.appendChild(ttsAudioEl);
  }
}

function startStreamPlayback() {
  stopPlayback();
  ensureAudioCtx();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  streamNextTime = audioCtx.currentTime + 0.05; // Small initial buffer
  speakingStartedAt = Date.now();
  setState('speaking');
}

function queueAudioChunk(base64Pcm) {
  ensureAudioCtx();

  // Decode base64 -> Int16 PCM -> Float32
  const bin = atob(base64Pcm);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

  const audioBuffer = audioCtx.createBuffer(1, float32.length, streamSampleRate);
  audioBuffer.getChannelData(0).set(float32);

  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ttsOut);
  source.connect(analyser);

  // Schedule gapless playback
  const startAt = Math.max(streamNextTime, audioCtx.currentTime);
  source.start(startAt);
  streamNextTime = startAt + audioBuffer.duration;

  streamSources.push(source);

  source.onended = () => {
    const idx = streamSources.indexOf(source);
    if (idx !== -1) streamSources.splice(idx, 1);
    if (streamSources.length === 0 && state === 'speaking') goListening();
  };
}

// ── UI ──
function addMessage(role, text, meta) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `${text}${meta ? `<div class="meta">${meta}</div>` : ''}`;
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

cameraToggle.addEventListener('click', () => {
  cameraEnabled = !cameraEnabled;
  cameraToggle.classList.toggle('active', cameraEnabled);
  cameraToggle.textContent = cameraEnabled ? 'Camera On' : 'Camera Off';
  video.style.opacity = cameraEnabled ? 1 : 0.3;
});

// ── Init ──
async function init() {
  initWaveformCanvas();
  window.addEventListener('resize', initWaveformCanvas);

  await startCamera();
  connect();

  myvad = await vad.MicVAD.new({
    getStream: async () => new MediaStream(mediaStream.getAudioTracks()),
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.25,
    // Short redemption for fast turn-taking; the server's smart-turn
    // classifier catches the cases where the user was merely pausing.
    redemptionMs: 200,
    minSpeechMs: 300,
    preSpeechPadMs: 300,
    onSpeechStart: handleSpeechStart,
    onSpeechEnd: handleSpeechEnd,
    onVADMisfire: () => {
      // No speech-end will follow — let the next real utterance send a
      // fresh frame instead of pointing the model at this stale one.
      framePrefetched = false;
      speechActive = false;
      if (serverHoldingAudio) startFlushTimer();
      console.log('VAD misfire (too short)');
    },
    onFrameProcessed: (probs, frame) => handleVadFrame(probs, Float32Array.from(frame)),
    onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
    baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
  });

  myvad.start();

  // Init audio context on first user gesture for mic visualizer
  // (browsers require user interaction to create AudioContext)
  const initAudio = () => {
    ensureAudioCtx();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    if (ttsAudioEl) ttsAudioEl.play().catch(() => {});
    document.removeEventListener('click', initAudio);
    document.removeEventListener('keydown', initAudio);
  };
  document.addEventListener('click', initAudio);
  document.addEventListener('keydown', initAudio);
  // Also try immediately (may work if page had prior interaction)
  ensureAudioCtx();

  setState('listening');
  drawWaveform();

  console.log('VAD initialized and listening');
}

init();
