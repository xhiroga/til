// undifined
class English {
    constructor() {
        setTimeout(
            () => { this.greeting = 'Hello' }, 5000
        )
    }
}

lang = new English()
console.log(lang.greeting)

// async constructor() {
//           ^^^^^^^^^^^
//         SyntaxError: Class constructor may not be an async method

class Japanese {
    async constructor() {
        setTimeout(
            () => { this.greeting = 'こんにちは' }, 5000
        )
    }
}

lang = await new English()
console.log(lang.greeting)