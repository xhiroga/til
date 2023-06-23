# Vercel KV

## Debug

```shell
export $(cat .env)
redis-cli --tls -u ${KV_URL}
```

## Develop

```shell
pnpm dev
open localhost:3000/api/user
```

## References

- [Vercel KV Quickstart | Vercel Docs](https://vercel.com/docs/storage/vercel-kv/quickstart#)
