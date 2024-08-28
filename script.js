async function setupCamera() {
    const video = document.getElementById('video');
    video.width = 640;
    video.height = 480;

    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });

    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function detectPoses(net, video, canvas) {
    const ctx = canvas.getContext('2d');
    canvas.width = video.width;
    canvas.height = video.height;

    async function poseDetectionFrame() {
        const pose = await net.estimateSinglePose(video, {
            flipHorizontal: false
        });

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (pose) {
            pose.keypoints.forEach(({ position }) => {
                ctx.beginPath();
                ctx.arc(position.x, position.y, 5, 0, 2 * Math.PI);
                ctx.fillStyle = 'red';
                ctx.fill();
            });
        }

        requestAnimationFrame(poseDetectionFrame);
    }

    poseDetectionFrame();
}

async function main() {
    const video = await setupCamera();
    video.play();

    const net = await posenet.load();
    const canvas = document.getElementById('output');
    detectPoses(net, video, canvas);
}

main();
