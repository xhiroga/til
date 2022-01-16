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

export type ApiEnvelop = {
  kind: Kind;
  apiVersion: string;
};
export type ApiListEnvelop = ApiEnvelop & {
  metadata: {
    resourceVersion: string;
  };
  items: [];
};

const getResourceFromRequest = (request: ApiEnvelop & Resource) => {
  const { kind, apiVersion, ...payload } = request;
  return payload;
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

const getItemResponse = (
  kind: ItemKind,
  apiVersion: string,
  resource: Resource
) => {
  return {
    kind: kind,
    apiVersion: apiVersion,
    ...resource,
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
  getResourceFromRequest,
  getApiGroupResponse,
  getApiGroupListResponse,
  getApiResourceListResponse,
  getItemResponse,
  getListResponse,
};
