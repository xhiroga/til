# CAST関数
値のデータタイプを変換するための関数。

## Double <> Decimal
```sql
SELECT CAST('123.456' AS DOUBLE)
FROM SYSIBM.SYSDUMMY1; -- DB2におけるダミーテーブル。今回はテーブルは何でもいいので、このように指定しておく。
```