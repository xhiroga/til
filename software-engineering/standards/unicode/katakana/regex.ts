import { Pattern } from "https://deno.land/x/regexbuilder@1.6.3/mod.ts";

const regs = [
    '^[ァ-ン]+$',  // https://gist.github.com/terrancesnyder/1345094
    '^[ァ-ヾ]+$',  // https://gist.github.com/terrancesnyder/1345094?permalink_comment_id=2855970#gistcomment-2855970
    '^[゙-゜゠-ヿㇰ-ㇿ]+$',
]

const test = (reg: string) => {
    console.log(reg);
    const pattern = Pattern.new().template("reg").vars({ reg }).build();
    console.log(pattern.test('アイウエオ'));
    console.log(pattern.test('ガラス'));    // ガ ではなく、基底文字「カ」と結合文字「゙」の組み合わせ
    console.log(pattern.test('イスヾ'));    // 踊り字
    console.log(pattern.test('キャサリン・ゼタ゠ジョーンズ')); // [ダブルハイフン - Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%80%E3%83%96%E3%83%AB%E3%83%8F%E3%82%A4%E3%83%95%E3%83%B3)
    console.log(pattern.test('ボクハトテモキミニサイカイスルヿハデキヌトオモウ'));  // [ヿ - Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%BF)
    console.log(pattern.test('ピㇼカコタン'));  // [Unicode 片仮名拡張 - CyberLibrarian](https://www.asahi-net.or.jp/~ax2s-kmtn/ref/unicode/u31f0.html)
    console.log(pattern.test('シンカ゚ーソンク゚ライター'));  // [か゜ - Wikipedia](https://ja.wikipedia.org/wiki/%E3%81%8B%E3%82%9C)
}

regs.forEach(test);
