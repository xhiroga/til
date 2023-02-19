# Interoperating with Node and NPM

## Run

```shell
deno run --unstable --allow-read index.ts
# Denoのキャッシュディレクトリ ($HOME/Library/Caches/deno/npm/...) の読み取り権限が必要ということらしい。
```

## References

- [Interoperating with Node and NPM](https://deno.land/manual@v1.26.1/node)
- [Deno 1.25.2 doesn't accept `--compat` flag any more. · Issue #15854 · denoland/deno · GitHub](https://github.com/denoland/deno/issues/15854)
