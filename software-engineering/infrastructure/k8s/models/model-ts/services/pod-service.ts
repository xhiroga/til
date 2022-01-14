import { podMap } from "../db/reources-map.ts";
import { Pod } from "../definitions/resources.ts";

const getPods = (namespace?: string) => {
  if (namespace === undefined) {
    return podMap.flatMap((map) => map.pods);
  } else {
    return podMap.filter((map) => map.namespace === namespace)[0].pods;
  }
};
const createPod = (namespace: string, pod: Pod) => {
  const pods = podMap.filter((map) => map.namespace === namespace)[0].pods;
  podMap
    .filter((map) => map.namespace !== namespace)
    .push({
      namespace: namespace,
      pods: pods,
    });
  return pod;
};

export { getPods, createPod };
