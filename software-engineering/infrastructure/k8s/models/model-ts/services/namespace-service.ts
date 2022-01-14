const getNameSpaces = () => {
  return [
    {
      metadata: {
        name: "default",
        uid: "69512b97-5c4d-4b6f-ac80-3dd4c92c09c9",
        resourceVersion: "19",
        creationTimestamp: "2021-05-03T02:10:48Z",
        managedFields: [
          {
            manager: "k3s",
            operation: "Update",
            apiVersion: "v1",
            time: "2021-05-03T02:10:48Z",
            fieldsType: "FieldsV1",
            fieldsV1: {
              "f:status": {
                "f:phase": {},
              },
            },
          },
        ],
      },
      spec: {
        finalizers: ["kubernetes"],
      },
      status: {
        phase: "Active",
      },
    },
  ];
};
export { getNameSpaces };
