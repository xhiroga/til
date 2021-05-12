#!/usr/bin/env deno run

// [英辞郎のテキストファイル](https://booth.pm/ja/items/777563)をログファイルに見立てて処理する。
// cat EIJIRO-1446.TXT > deno grep.ts calcium

import { readLines } from "https://deno.land/std/io/mod.ts";

const keyword = Deno.args[0]
console.log(`keyword: ${keyword}`)

for await (let line of readLines(Deno.stdin)) {
    if (line.indexOf(keyword) >= 0) {
        console.log(line)
    }
}
