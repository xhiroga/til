const fs = require('fs')
const Promise = require('bluebird')
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const ffmpeg = require('fluent-ffmpeg')
ffmpeg.setFfmpegPath(ffmpegPath);

const unlink = Promise.promisify(fs.unlink)

const toWav = async (pcmPath) => {
    const destPath = pcmPath + '.wav'
    return new Promise((resolve, reject) => {
        ffmpeg()
            .input(pcmPath)
            .inputOptions(['-ac 1', '-ar 16000'])
            .inputFormat('s16be')
            .output(destPath)
            .on('end', () => {
                console.log(destPath)
                unlink(pcmPath).then(resolve)
            })
            .on('error', reject)
            .run()
    })
}

toWav('./pcm/file/path')

// Reference
// https://blog.leko.jp/post/voiceloid-like-text2speech/