const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();

            // Start recognition only after camera fully loads
            setInterval(sendFrame, 2000);
        };

    } catch (err) {
        console.error("Camera Error:", err);
    }
}

function sendFrame() {

    if (video.videoWidth === 0 || video.videoHeight === 0) {
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const image = canvas.toDataURL('image/jpeg', 0.8);

    if (!image || image.length < 100) {
        return;
    }

    fetch('/recognize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: image
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('result').innerText = data.result;
    })
    .catch(err => console.log(err));
}

startCamera();