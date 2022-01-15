import { DictionaryService } from "./DictionaryService.ts";

export const myDictionary: DictionaryService = {
  "d:dictionary": {
    "@xmlns": "http://www.w3.org/1999/xhtml",
    "@xmlns:d": "http://www.apple.com/DTDs/DictionaryService-1.0.rng",
    "d:entry": [
      {
        "@id": "smb",
        "@d:title": "SMB",
        "d:index": [{
          "@d:value": "smb",
        }],
        "h1": "SMB",
        "p":
          "ファイル転送プロトコルのデファクトスタンダード。iOSのファイルアプリで「サーバへ接続」する際に利用可能なプロトコルのうち、AFPではない方。",
      },
      {
        "@id": "webdav",
        "@d:title": "WebDAV",
        "d:index": [{
          "@d:value": "webdav",
        }],
        "h1": "WebDAV",
        "p":
          "HTTPを拡張したファイル転送プロトコル。WindowsとOS Xはいずれも、Web共有フォルダーのプロトコルとしてサポートしている。",
      },
    ],
  },
};
