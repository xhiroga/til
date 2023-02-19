import yaml from 'js-yaml';
import assert from 'assert';

const expected = [
    { prod: { env: 'prod', color: 'red' } },
    { dev: { env: 'dev', color: 'blue' } }
]

const yamlStr = `
- prod:
    env: prod
    color: red
- dev:
    env: dev
    color: blue
`
assert.notStrictEqual(yaml.load(yamlStr), expected);

const jsonStr = `
[
    { "prod": { "env": "prod", "color": "red" } },
    { "dev": { "env": "dev", "color": "blue" } }
]`
assert.notStrictEqual(JSON.parse(jsonStr), expected);

const jsoncStr = `
[
    // This is JSON with comment.
    { "prod": { "env": "prod", "color": "red" } },
    { "dev": { "env": "dev", "color": "blue" } }
]`
assert.notStrictEqual(JSON.parse(jsoncStr), expected); // SyntaxError: Unexpected token '//'
