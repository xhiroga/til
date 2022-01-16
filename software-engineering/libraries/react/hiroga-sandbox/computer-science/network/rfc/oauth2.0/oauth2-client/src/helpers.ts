export const getEnv = (varName: string) => {
    const value = process.env[varName]
    if(value === undefined){
        throw Error(`Environment variable ${varName} was not set.`)
    }
    return value
}

export const base64Encode = (value: string) => Buffer.from(value).toString('base64');
