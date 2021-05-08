const apiVersions = {
  kind: "APIVersions",
  versions: ["v1"],
  serverAddressByClientCIDRs: [
    {
      clientCIDR: "0.0.0.0/0",
      serverAddress: "172.29.0.2:6443",
    },
  ],
};
export { apiVersions };
