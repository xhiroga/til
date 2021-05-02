export type ApiResponse = {
  kind: string;
  apiVersion: string;
  metadata: {
    selflink?: string;
    resourceVersion: string;
  };
  items: NameSpace[] | Node[];
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

export type Pod = {
  container: any;
  storageVolume: any;
};

export type Service = {};

export type Deployment = {};
