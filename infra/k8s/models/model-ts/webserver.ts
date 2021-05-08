import { Application, Router } from "https://deno.land/x/oak/mod.ts";
import { apiGroupList } from "./models/api-group-list.ts";
import { apiVersions } from "./models/api-versions.ts";
import { getApiResourceList } from "./services/api-resource-service.ts";
import { getApiResourceListResponse } from "./web/api-response.ts";

const apiRouter = new Router()
  .get("/api", (ctx) => {
    ctx.response.body = apiVersions;
  })
  .get("/api/:version", (ctx) => {
    const version = ctx.params.version!;
    ctx.response.body = getApiResourceListResponse(
      version,
      version,
      getApiResourceList(version)
    );
  });
const apisRouter = new Router()
  .get("/apis", (ctx) => {
    // kubectl は 取得した apis それぞれに対して API Resourceの取得を試みるため、本モデルでは apps 以外のAPIを省いた。
    ctx.response.body = {
      kind: "APIGroupList",
      apiVersion: "v1",
      groups: apiGroupList,
    };
  })
  .get("/apis/:name/:version", (ctx) => {
    const version = ctx.params.version!;
    const groupVersion = `${ctx.params.name!}/${version}`;
    ctx.response.body = getApiResourceListResponse(
      version,
      groupVersion,
      getApiResourceList(groupVersion)
    );
  });

await new Application()
  .use(apiRouter.routes())
  .use(apisRouter.routes())
  .listen({ port: 8000 });
