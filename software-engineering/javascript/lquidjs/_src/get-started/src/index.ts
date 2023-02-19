import { Liquid } from 'liquidjs'
const engine = new Liquid()
const tpl = engine.parse('Welcome to {{v}}!')
engine.render(tpl, { v: "Liquid" }).then(console.log)
// Outputs "Welcome to Liquid!"

const obj = { "name": "liquid" }
engine.render(tpl, { v: obj }).then(console.log)
// Welcome to [object Object]!

const ary = [{ "id": 123 }]
engine.render(tpl, { v: ary }).then(console.log)
// Welcome to [object Object]!
