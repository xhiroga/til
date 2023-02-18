import { stringify } from "https://deno.land/x/xml/mod.ts";
import { myDictionary } from "./my-dictionary.ts";

const myDictionaryXml = stringify(myDictionary);

Deno.writeTextFile(
  Deno.env.get("DICT_SRC_PATH") || "MyDictionary.xml",
  `<?xml version="1.0" encoding="UTF-8"?>
${myDictionaryXml}`,
);
