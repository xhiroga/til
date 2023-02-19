# wasm.supabase.com

## Connecting to PostgreSQL

```psql
ALTER USER postgres WITH PASSWORD 'password';
```

```shell
export PORT=${IN_THE_STATUS_BAR}
psql postgres://postgres@proxy.wasm.supabase.com:${PORT}
```

```psql
select * from pg_database;  # list databases
select * from pg_user;  # list users
select * from pg_stat_activity; # connections
quit;
```
