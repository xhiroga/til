const encoder = new TextEncoder()
const salt1 = "This program is written by Hiroaki Ogasawara"
const salt2 = "who live in Asakusa"
const salt3 = "now I am hungry"

const wordToDeg = (word) => {
    const utf8array = encoder.encode(word)
    return Math.floor(utf8array[0] * utf8array[1] % 360)
}

const wordToRgb = (word) => {
    const utf8array = encoder.encode(word)
    return { r: utf8array[0], g: utf8array[1], b: utf8array[2] }
}

const linearGradient = (deg, r, g, b) => {
    return `linear-gradient(${deg}deg, rgba(${r},${g},${b},.8), rgba(${r},${g},${b},0) 70.71%)`
}

const shuffle = (word) => {
    var shuffledWord = '';
    word = word.split('');
    while (word.length > 0) {
        shuffledWord += word.splice(word.length * Math.random() << 0, 1);
    }
    return shuffledWord;
}
const wordToLinear = (word) => {
    const shuffledWord = shuffle(word)
    const rgb = wordToRgb(shuffledWord)
    return linearGradient(wordToDeg(shuffledWord), rgb.r, rgb.g, rgb.b)
}

const paint = () => {
    let word = $("#yourWordInput").val()
    if (word.length === 0) {
        word = "無事故でチャリ通"
    }

    const linearGradients = `${wordToLinear(word + salt1)},${wordToLinear(word + salt2)},${wordToLinear(word + salt3)}`
    $("#yourwordInupted").text(word)
    $("#canvas").css("background", linearGradients)
};
