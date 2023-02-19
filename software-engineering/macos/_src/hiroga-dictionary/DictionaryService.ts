type Index = {
  "@d:value": string;
  "@d:title"?: string;
  "@d:anchor"?: string;
  "@d:yomi"?: string;
  "@d:parental-control"?: number;
  "@d:priority"?: number;
};

type Entry = {
  "@id": string;
  "@d:title": string;
  "@d:parental-control"?: number;
  "d:index": Index[];
  h1: string;
  p: string;
};

type Dictionary = {
  "@xmlns": string;
  "@xmlns:d": string;
  "d:entry": Entry[];
};

export type DictionaryService = {
  "d:dictionary": Dictionary;
};
