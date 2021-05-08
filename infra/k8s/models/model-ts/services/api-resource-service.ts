import { apiResourceListMap } from "../models/api-resource-list-map.ts";
const getApiResourceList = (groupVersion: string) => {
  return apiResourceListMap.filter(
    (map) => map.groupVersion === groupVersion
  )[0].apiResourceList;
};
export { getApiResourceList };
