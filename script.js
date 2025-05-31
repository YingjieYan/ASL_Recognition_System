let interval = null;

function startCaption() {
    if (!interval) {
        interval = setInterval(() => {
            fetch('/get_text')
                .then(res => res.json())
                .then(data => {
                    document.getElementById("subtitle").innerText = data.text;
                });
        }, 1000);
    }
}

function clearCaption() {
    fetch('/clear_text')
        .then(() => {
            document.getElementById("subtitle").innerText = "";
        });
}

function speak() {
    const text = document.getElementById("subtitle").innerText;
    const utterance = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(utterance);
}
