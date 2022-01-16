// Interface
// この { first: string, last: string } をオブジェクト型、記法をオブジェクト型リテラルと呼ぶ。
export interface Name {
    first: string,
    last: string
}

export const name: Name = {
    first: 'Ogasawara',
    last: 'Hiroaki'
}

// Inline Annotation
export const name2: {
    first: string,
    last: string
} = {
    first: '信長',
    last: '織田'
}

// Intersection Type
function extend<T, U>(first: T, second: U): T & U {
    return { ...first, ...second };
}

interface Bird {
    wing: string
}

interface Lion {
    tusk: string
}

const griffon: Bird & Lion = extend({ wing: "large" }, { tusk: "looks strong" });

// Type Alias (is OK to Union type and Intersection Type) (but cannot have hierarchy)
type Griffon = Bird & Lion
const myGriffon: Griffon = { wing: "sharp", tusk: "also sharp" }

// Ambient Declaration
// TypeScriptさん、あなたは知らないかもしれないけど、私の環境では動くんですよ！の意味(HTMLで別のjsファイルを読み込んでいるとか)
// 実際のプロジェクトでは、 global.d.ts などのファイルに隔離しておく。
declare function TextEncoder(): void;

// Interfaceは拡張できる
interface Pikachu {
    thunderbolt: number
}

interface Pikachu {
    voltTackle: number
}

const myPikachu = { thunderbolt: 90, voltTackle: 120 }

// Function Declaration
// 自分でTypeScriptのコードを書く場合にはインラインの型アノテーションで事足りるはずだから、これは型定義ファイルを外部に書くときのTipsかな。
type LongHand = {
    (a: number): number;
};

type ShortHand = (a: number) => number;

// LongHandの関数定義って初めて見た...
type LongHandAllowsOverloadDeclarations = {
    (a: number): number;
    (a: string): string;
};

// Type Assertion
// Type Annotationとは違い、変数宣言時にプロパティを追加し忘れてもコンパイルエラーにならない。JavaScriptをTypeScriptに移行するための機能と捉えたほうが良さそう。
// もしくは、ビジネスロジックによって変数のサブタイプが決まる場合。
interface Foo {
    bar: number;
    bas: string;
}
var foo = {} as Foo;    // var foo: Foo だと、Type '{}' is missing the following properties from type 'Foo': bar, basts(2739)
foo.bar = 123;          // 型アサーションがない場合、Property 'bar' does not exist on type '{}'.ts(2339)
foo.bas = 'hello';      // 型アサーションがない場合、Property 'bar' does not exist on type '{}'.ts(2339)


// Freshnessについて...
// ある型の変数にオブジェクトリテラルを代入する際に、プロパティの有無を厳密にチェックする機能。
// ...多分本当は厳密にチェックしたかったんだろうけど、変数の場合はどこでmutableな変更が入っているかわからないしオブジェクトリテラルに限定したんだろうな。

function logName(something: { name: String }) {
    console.log(something.name)
}
const person = { name: "Hideyoshi", age: "20" }
logName(person)
// logName({ name: "Ieyasu", age: "20" }) // Argument of type '{ name: string; age: string; }' is not assignable to parameter of type '{ name: String; }'. Object literal may only specify known properties, and 'age' does not exist in type '{ name: String; }'.ts(2345)

let x: string | number
x = 'x'
// console.log(x.substr(1))     // Property 'substr' does not exist on type 'string | number'. Property 'substr' does not exist on type 'number'.ts(2339)
if (typeof x === 'string') {
    console.log(x.substr(1))
}

class Foo {
    foo = 123
    common = '123'
}

// (ストリング)リテラル型
const A: 'a' = 'a'
console.log(A)

let B: 'b'
// console.log(b)   // このように代入前に使用できるように見えるが、コンパイルエラーになる。

function toBeOrNotToBe(how: 'toBe' | 'notToBe') {
    console.log(how)
}
const ibm = { how: 'toBe' }
const apple = { how: 'toC' }
// toBeOrNotToBe(ibm.how) // Argument of type 'string' is not assignable to parameter of type '"toBe" | "notToBe"'.ts(2345)
toBeOrNotToBe(ibm.how as 'toBe') // OK. もちろん宣言時にinlineで型アノテーションしたほうがお行儀いいはず。

// deep-dive には既存の文字列の配列から文字列リテラルの合成型を生成する方法が載っているが、複雑でよく分からない...

// never型
function neverEndingStory(): never {
    throw Error("The story will be continue!")
}
