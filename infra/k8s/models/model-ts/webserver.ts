import { Application, Router } from "https://deno.land/x/oak/mod.ts";
import {
  v1ApiResourceList,
  appsV1ApiResourceList,
} from "./db/api-resource-list.ts";

import { getNameSpaces } from "./services/namespace-service.ts";
import { apiVersions } from "./db/api-versions.ts";
import { openApiV2 } from "./db/openapi-v2.ts";
import { paths } from "./db/paths.ts";
import { createPod, getPods } from "./services/pod-service.ts";
import {
  getApiGroupResponse,
  getApiGroupListResponse,
  getApiResourceListResponse,
  getListResponse,
} from "./web/api-helper.ts";

const podsRouter = new Router()
  .get("/", (ctx) => {
    ctx.response.body = getListResponse(
      "PodList",
      ctx.params.appsVersion!,
      getPods(ctx.params.namespace)
    );
  })
  .post("/", (ctx) => {
    console.log(ctx.request.body());
    createPod(ctx.request.body());
  });
const serviceAccountsRouter = new Router();
const servicesRouter = new Router();
const withV1ResourcesRouter = (router: Router) => {
  return router
    .use("/pods", podsRouter.routes())
    .use("/serviceaccounts", serviceAccountsRouter.routes())
    .use("/services", servicesRouter.routes());
};
const withAppsV1ResourcesRouter = (router: Router) => {
  return router;
};

const getNamespaceRouter = (withChildren: (router: Router) => Router) => {
  return new Router()
    .get("/", (ctx) => {
      ctx.response.body = getListResponse(
        "NamespaceList",
        ctx.params.version!, // 例えば /api/v1 と /apis/apps/v2 が共存した場合、どちらを使うのだろうか。
        getNameSpaces()
      );
    })
    .use(
      "/:namespace",
      withChildren(
        new Router()
          .get("/", (ctx) => {})
          .post("/", (ctx) => {})
          .patch("/", (ctx) => {})
          .delete("/", (ctx) => {})
      ).routes()
    );
};

const apiRouter = new Router()
  .get("/", (ctx) => {
    ctx.response.body = { kind: "APIVersions", ...apiVersions };
  })
  .use(
    "/:version",
    new Router()
      .get("/", (ctx) => {
        const version = ctx.params.version!;
        ctx.response.body = getApiResourceListResponse(
          version,
          v1ApiResourceList
        );
      })
      .use("/namespaces", getNamespaceRouter(withV1ResourcesRouter).routes())
      .routes()
  );

const appsApiRouter = new Router()
  .get("/", (ctx) => {
    ctx.response.body = getApiGroupResponse();
  })
  .use(
    "/:appsVersion",
    new Router()
      .get("/", (ctx) => {
        ctx.response.body = getApiResourceListResponse(
          `apps/${ctx.params.appsVersion}`,
          appsV1ApiResourceList // インチキ
        );
      })
      .use(
        "/namespaces",
        getNamespaceRouter(withAppsV1ResourcesRouter).routes()
      )
      .routes()
  );

const apisRouter = new Router()
  .get("/", (ctx) => {
    // kubectl は 取得した apis それぞれに対して API Resourceの取得を試みるため、本モデルでは apps 以外のAPIを省いた。
    ctx.response.body = getApiGroupListResponse();
  })
  .use("/apps", appsApiRouter.routes());

const routeRouter = new Router()
  .get("/", (ctx) => (ctx.response.body = { paths: paths }))
  .use("/api", apiRouter.routes())
  .use("/apis", apisRouter.routes())
  .get("/openapi/v2", (ctx) => {
    // とりあえず生やしてみたものの、OpenAPIのチェックを通せなかった。--validate=false で実行する。
    ctx.response.headers.append("content-type", "text/plain; charset=utf-8");
    ctx.response.body = openApiV2;
  });

await new Application().use(routeRouter.routes()).listen({ port: 8000 });
