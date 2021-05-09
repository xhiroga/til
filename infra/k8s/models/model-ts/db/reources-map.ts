import { Pod } from "../definitions/resources.ts";

const podMap: { namespace: string; pods: Pod[] }[] = [
  { namespace: "default", pods: [] },
];

export { podMap };
