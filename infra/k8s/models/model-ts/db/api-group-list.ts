import { ApiGroup } from "../definitions/resources.ts";

const apiGroupList: ApiGroup[] = [
  {
    name: "apps",
    versions: [
      {
        groupVersion: "apps/v1",
        version: "v1",
      },
    ],
    preferredVersion: {
      groupVersion: "apps/v1",
      version: "v1",
    },
  },
];
export { apiGroupList };
