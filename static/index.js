const canvas = document.getElementById('drawing-board');
const toolbar = document.getElementById('toolbar');
const ctx = canvas.getContext('2d');

const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

canvas.width = window.innerWidth - canvasOffsetX;
canvas.height = window.innerHeight - canvasOffsetY;

let isPainting = false;
let lineWidth = 5;
let startX;
let startY;
let currX;
let currY;
coordinatesElement = document.getElementById('coordinates');
let isMousePressed = false;

toolbar.addEventListener('click', e => {
    if (e.target.id === 'clear') {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
});

toolbar.addEventListener('change', e => {
    if(e.target.id === 'stroke') {
        ctx.strokeStyle = e.target.value;
    }

    if(e.target.id === 'lineWidth') {
        lineWidth = e.target.value;
    }
    
});

const draw = (e) => {
    if(!isPainting) {
        return;
    }

    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';

    ctx.lineTo(e.clientX - canvasOffsetX, e.clientY);
    ctx.stroke();
}

canvas.addEventListener('mousedown', (e) => {  
    isMousePressed = true;
    startX = e.clientX;
    startY = e.clientY;
    isPainting = true;
});

canvas.addEventListener('mouseup', e => {
    isMousePressed = false;

    isPainting = false;
    ctx.stroke();
    ctx.beginPath();
});

document.addEventListener('mousemove', function (event) {
    if (isMousePressed) {
        const x = event.clientX;
        const y = event.clientY;
        const coordinates = `(${x}, ${y})`;
        coordinatesElement.textContent = coordinates;

        // Send the coordinates to the server using a fetch request
        fetch('/process_coordinates', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 'coordinates': coordinates })
        });
    }
});

canvas.addEventListener('mousemove', draw);
