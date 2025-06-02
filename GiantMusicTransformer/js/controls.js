// 控制按钮相关逻辑
// document.getElementById("initBtn").addEventListener("click", async () => {
//     try {
//         msg = await loadModel();
//         alert("Model initialized successfully!");
//         console.log(msg);
//     } catch (error) {
//         alert("Failed to initialize model: " + error.message);
//     }
// });

async function initControls(){
    try {
        msg = await loadModel();
        alert("Model initialized successfully!");
        console.log(msg);
    } catch (error) {
        alert("Failed to initialize model: " + error.message);
    }
}