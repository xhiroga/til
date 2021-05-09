export type Resource = APIResource | Pod | ReplicaSet | Deployment;
export type Kind =
  // v1
  | "Pod"
  | "PodAttachOptions"
  | "PodExecOptions"
  | "PodProxyOptions"
  | "PodTemplate"
  | "Binding"
  | "ComponentStatus"
  | "ConfigMap"
  | "Endpoints"
  | "Event"
  | "Eviction"
  | "LimitRange"
  | "Namespace"
  | "NodeProxyOptions"
  | "Node"
  | "PersistentVolumeClaim"
  | "PersistentVolume"
  | "PodPortForwardOptions"
  | "ReplicationController"
  | "ResourceQuota"
  | "Scale"
  | "Secret"
  | "Service"
  | "ServiceAccount"
  | "ServiceProxyOptions"
  | "TokenRequest"
  // apps/v1
  | "ControllerRevision"
  | "DaemonSet"
  | "ReplicaSet"
  | "Deployment"
  | "StatefulSet";

export type Verb =
  | "create"
  | "delete"
  | "deletecollection"
  | "get"
  | "list"
  | "patch"
  | "update"
  | "watch";

export type ApiGroup = {
  name: string;
  versions: ApiGroupVersion[];
  preferredVersion: ApiGroupVersion;
};
type ApiGroupVersion = {
  groupVersion: string;
  version: string;
};

export type APIResource = {
  name: string;
  singularName: string;
  namespaced: boolean;
  group?: string;
  version?: string;
  kind: Kind;
  verbs: Verb[];
  shortNames?: string[];
  categories?: string[];
  storageVersionHash?: string;
};

export type Container = {
  image: string;
  name: string;
  ports: [];
};

export type Pod = {
  metadata: any;
  spec: {
    containers: Container[];
  };
  storageVolume: any;
};

const pods: Pod[] = [];
export { pods };

export type ReplicaSet = {
  metadata: any;
  spec: {
    replicas: number;
    selector: any;
    template: Pod;
  };
  status: any;
};

export type Deployment = {
  strategy: any;
} & ReplicaSet;

export type Network = {};

export type Volume = {};

export type NameSpace = {};

export type Node = {
  metadata: NodeMetadata;
  spec: {
    podCIDR: string;
    podCIDRs: string[];
    providerID: string;
  };
  status: {};
};

export type NodeMetadata = {
  name: string;
  selfLink: string;
  uid: string;
  resourceVersion: string;
  creationTimestamp: Date;
  finalizers: string[];
};

export type Worker = {};

export type Service = {};
