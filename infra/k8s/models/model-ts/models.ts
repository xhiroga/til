export type Kind = "Pod" | "PodList" | "ReplicaSet" | "ReplicaSetList";
export type ApiVersion = "v1" | "apps/v1";

export type ApiResponse = {
  kind: Kind;
  apiVersion: ApiVersion;
};
export type ApiListResponse = ApiResponse & {
  metadata: {
    resourceVersion: string;
  };
  items: [];
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
export type PodResponse = ApiResponse & Pod;
export type PodListResponse = ApiListResponse & {
  items: Pod[];
};

export type ReplicaSet = {
  metadata: any;
  spec: {
    replicas: number;
    selector: any;
    template: Pod;
  };
  status: any;
};
export type ReplicaSetResponse = ApiResponse & ReplicaSet;
export type ReplicaSetListResponse = ApiListResponse & {
  items: ReplicaSet[];
};

export type Deployment = {
  strategy: any;
} & ReplicaSet;
export type DeploymentResponse = ApiResponse & Deployment;
export type DeploymentListResponse = ApiListResponse & {
  items: Deployment[];
};

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
