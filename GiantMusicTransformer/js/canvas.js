// 与画布相关的逻辑
const drawCanvas = document.getElementById("drawCanvas");
const drawCtx = drawCanvas.getContext("2d");

let drawing = false;

drawCtx.lineWidth = 4;
drawCtx.lineCap = "round";
drawCtx.strokeStyle = "#000";

drawCanvas.addEventListener("mousedown", (e) => {
    drawing = true;
    const { x, y } = getMousePos(drawCanvas, e);
    drawCtx.beginPath();
    drawCtx.moveTo(x, y);
});

drawCanvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const { x, y } = getMousePos(drawCanvas, e);
    drawCtx.lineTo(x, y);
    drawCtx.stroke();
});

drawCanvas.addEventListener("mouseup", () => {
    drawing = false;
});

drawCanvas.addEventListener("mouseleave", () => {
    drawing = false;
});

document.getElementById("clearBtn").addEventListener("click", () => {
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
});