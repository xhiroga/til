import { APIResource, Kind, Resource } from "../definitions/resources.ts";
import { apiGroupList } from "../db/api-group-list.ts";

type ItemKind = "APIVersions" | "APIGroup" | "APIResource" | Kind;
type ListKind =
  | "APIGroupList"
  | "APIResourceList"
  | "NamespaceList"
  | "PodList"
  | "ReplicaSetList"
  | "DeploymentList";

export type ApiResponse = {
  kind: Kind;
  apiVersion: string;
};
export type ApiListResponse = ApiResponse & {
  metadata: {
    resourceVersion: string;
  };
  items: [];
};

const getApiGroupResponse = () => {
  return {
    kind: "APIGroup",
    apiVersion: "v1",
    ...apiGroupList.filter((group) => group.name === "apps")[0],
  };
};

const getApiGroupListResponse = () => {
  return {
    kind: "APIGroupList",
    apiVersion: "v1",
    groups: apiGroupList,
  };
};

const getApiResourceListResponse = (
  groupVersion: string,
  resources: APIResource[]
) => {
  return {
    kind: "APIResourceList",
    groupVersion: groupVersion,
    resources: resources,
  };
};

const getListResponse = (
  kind: ListKind,
  apiVersion: string,
  items: Resource[]
) => {
  return {
    kind: kind,
    apiVersion: apiVersion,
    items: items,
  };
};

export {
  getApiGroupResponse,
  getApiGroupListResponse,
  getApiResourceListResponse,
  getListResponse,
};
