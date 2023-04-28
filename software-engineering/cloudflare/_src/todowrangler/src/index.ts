export interface Env {
	TODO_WRANGLER: KVNamespace;
}

export default {
	async fetch(
		request: Request,
		env: Env,
		ctx: ExecutionContext
	): Promise<Response> {
		const value = await env.TODO_WRANGLER.get("test");

		if (value === null) {
			return new Response("Value not found", { status: 404 })
		}
		return new Response(value);
	},
};
