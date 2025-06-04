// MIDI相关逻辑
const fileInput = document.getElementById("fileInput");
const canvas = document.getElementById("pianoRoll");
const ctx = canvas.getContext("2d");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");

const halfStepHeight = 8;
const audioCtx = new AudioContext();
const instruments = {};
const playingNotes = [];
const ZOOM_STEP = 0.1;
const minWidth = 1400;
const minMaxTime = minWidth / 5 / halfStepHeight;

let midi = null;
let synth = null;
let isPlaying = false;
let startTime = 0;
let elapsedWhenPaused = 0;
let animationFrameId = null;
let zoomLevel = 1.0;
let maxTime = minMaxTime;

function applyCanvasScale() {
    const canvas = document.getElementById("pianoRoll");
    canvas.style.transform = `scale(${zoomLevel})`;
    canvas.style.transformOrigin = "center left";
}

function zoomIn() {
    zoomLevel = Math.min(zoomLevel + ZOOM_STEP, 10);
    applyCanvasScale();
}

function zoomOut() {
    zoomLevel = Math.max(zoomLevel - ZOOM_STEP, 0.2);
    applyCanvasScale();
}

async function loadModel() {
    console.log("Loading model and seed...");
    try {
        const response = await fetch("http://localhost:5000/load-model");
        console.log(response);
        if (!response.ok) {
            throw new Error("Failed to load model");
        }
        console.log("Model loaded successfully");
    } catch (error) {
        console.error("Error loading model:", error);
    }
    try {
        const response = await fetch("http://localhost:5000/load-seed-midi");
        console.log(response);
        if (!response.ok) {
            throw new Error("Failed to load seed");
        }
        console.log("seed loaded successfully");
    } catch (error) {
        console.error("Error loading seed:", error);
    }
}

async function runLocalModelAndDisplay() {
    const canvas = document.getElementById("drawCanvas");
    const imageData = canvas.toDataURL("image/png");

    // 显示加载指示器
    const loadingIndicator = document.getElementById("loadingIndicator");
    loadingIndicator.style.display = "block";

    try {
        const response = await fetch("http://localhost:5000/generate-midi", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ image: imageData })
        });

        console.log("after generation", response);
        if (!response.ok) {
            console.error("Failed to fetch MIDI from server");
            throw new Error("Failed to fetch MIDI from server");
        }

        const midiBlob = await response.blob();
        const arrayBuffer = await midiBlob.arrayBuffer();
        midi = new Midi(arrayBuffer);
        elapsedWhenPaused = 0;
        drawMidi(midi, 0);
    } catch (error) {
        console.error("Error:", error);
    } finally {
        // 隐藏加载指示器
        loadingIndicator.style.display = "none";
    }
}

fileInput.addEventListener("change", async (e) => {
    pauseBtn.click();
    elapsedWhenPaused = 0;
    isPlaying = false;
    Tone.Transport.stop();
    Tone.Transport.cancel();
    if (synth) synth.dispose();
    cancelAnimationFrame(animationFrameId);
    const file = e.target.files[0];
    if (!file) return;
    const arrayBuffer = await file.arrayBuffer();
    midi = new Midi(arrayBuffer);
    zoomLevel = 1.0;
    applyCanvasScale();
    drawMidi(midi);
});

function drawMidi(midi, currentTime = 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const notes = midi.tracks.flatMap(track => track.notes);
    const maxNote = 108, minNote = 21;
    const pitchRange = maxNote - minNote;

    maxTime = Math.max(...notes.map(n => n.time + n.duration));
    maxTime = Math.max(maxTime, minWidth / 5 / halfStepHeight);

    canvas.height = pitchRange * halfStepHeight;
    canvas.width = maxTime * halfStepHeight * 5;

    const programColorMap = {};
    let colorIndex = 0;
    const colorPalette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f1c40f",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e"
    ];

    midi.tracks.forEach((track, tIdx) => {
        const program = track.instrument.number ?? tIdx;
        const color = programColorMap[program] ?? (programColorMap[program] = colorPalette[colorIndex++ % colorPalette.length]);

        for (const note of track.notes) {
            const x = (note.time / maxTime) * canvas.width;
            const w = (note.duration / maxTime) * canvas.width;
            const y = (canvas.height - ((note.midi - minNote) / pitchRange) * canvas.height);
            const h = 6;
            ctx.fillStyle = color;
            ctx.fillRect(x, y, w, h);
        }
    });

    // Draw red time line
    const lineX = (currentTime / maxTime) * canvas.width;
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(lineX, 0);
    ctx.lineTo(lineX, canvas.height);
    ctx.stroke();
}

function schedulePlayback() {
    synth = new Tone.PolySynth().toDestination();
    const now = Tone.now();

    midi.tracks.forEach(track => {
        track.notes.forEach(note => {
            if (note.time >= elapsedWhenPaused && note.duration > 0) {
                synth.triggerAttackRelease(
                    note.name,
                    note.duration,
                    now + (note.time - elapsedWhenPaused),
                    note.velocity
                );
            }
        });
    });
}

function animate() {
    if (!isPlaying) return;

    const now = Tone.now();
    const currentTime = now - startTime + elapsedWhenPaused;

    drawMidi(midi, currentTime);

    animationFrameId = requestAnimationFrame(animate);
}

playBtn.addEventListener("click", async () => {
    if (!midi) return;

    await Tone.start();

    if (!isPlaying) {
        isPlaying = true;
        startTime = Tone.now();
        Tone.Transport.start("+0.1", elapsedWhenPaused);
        schedulePlayback();
        animate();
    }
});

pauseBtn.addEventListener("click", () => {
    if (isPlaying) {
        isPlaying = false;
        Tone.Transport.pause();
        elapsedWhenPaused += Tone.now() - startTime;
        if (synth) synth.dispose();
        cancelAnimationFrame(animationFrameId);
    }
});

canvas.addEventListener("click", (e) => {
    if (!midi) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / zoomLevel;
    const canvasWidth = canvas.width;
    const targetTime = (clickX / canvasWidth) * maxTime;

    // Stop and reset
    isPlaying = false;
    Tone.Transport.stop();
    Tone.Transport.cancel();
    if (synth) synth.dispose();
    cancelAnimationFrame(animationFrameId);

    elapsedWhenPaused = targetTime;
    drawMidi(midi, elapsedWhenPaused);
});

document.getElementById("midi-input-button").addEventListener("click", function () {
    document.getElementById("fileInput").click();
});