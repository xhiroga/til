import { podMap } from "../db/reources-map.ts";

const getPods = (namespace?: string) => {
  if (namespace === undefined) {
    return podMap.flatMap((map) => map.pods);
  } else {
    return podMap.filter((map) => map.namespace === namespace)[0].pods;
  }
};
const createPod = (any: any) => {
  console.log(any);
};

export { getPods, createPod };
