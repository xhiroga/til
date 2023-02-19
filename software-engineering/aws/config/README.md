# AWS Config Queries

## EIPs not associated with ENI

```sql
SELECT
  *,
  tags,
  relationships
WHERE
  resourceType = 'AWS::EC2::EIP'
  AND relationships.resourceId NOT LIKE 'eni-%'
```
