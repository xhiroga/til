export const getTwitterId = (body: string) => {
    // なぜか () がマッチしない。bunのランタイムには正規表現のエスケープに関するバグがあるのだろうか？
    const res = body.match(/### Twitter ID .+\n\n(\w{1,15})\n\n/)
    return res ? res[1] : null
}

export const getSessionAbstract = (body: string) => {
    // カタカナもマッチしないようだ...
    const res = body.match(/### .+\n\n(.*)\n\n### .+\n\n.*\n\n### .+\n\n.*\n\n### .+\n\n.*\n\n### .+\n\n.*\n\n### .+\n\n.*$/)
    return res ? res[1] : null
}

export const getSessionHosoku = (body: string) => {
    const res = body.match(/### .+\n\n(.*)\n\n### .+\n\n.*\n\n### .+\n\n.*\n\n### .+\n\n.*\n\n### .+\n\n.*$/)
    return res ? res[1] : null
}

export const getToudanshaStartupPhase = (body: string) => {
    const res = body.match(/### .+\n\n(.*)\n\n### .+\n\n.*\n\n### .+\n\n.*\n\n### .+\n\n.*$/)
    return res ? res[1] : null
}

export const getJukoushaStartupPhase = (body: string) => {
    const res = body.match(/### .+\n\n(.*)\n\n### .+\n\n.*\n\n### .+\n\n.*$/)
    return res ? res[1] : null
}

export const getSessionTopic = (body: string) => {
    const res = body.match(/### .+\n\n(.*)\n\n### .+\n\n.*$/)
    return res ? res[1] : null
}

export const getSessionFormat = (body: string) => {
    const res = body.match(/### .+\n\n(.*)$/)
    return res ? res[1] : null
}
