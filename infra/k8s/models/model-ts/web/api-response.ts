const getApiResourceListResponse = (
  apiVersion: string,
  groupVersion: string,
  resources: any[]
) => {
  return {
    kind: "APIResourceList",
    apiVersion: apiVersion,
    groupVersion: groupVersion,
    resources: resources,
  };
};
export { getApiResourceListResponse };
