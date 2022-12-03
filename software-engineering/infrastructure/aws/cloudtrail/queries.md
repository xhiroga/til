# Sample Queries

## Detect AWS SSO AdministratorAccess

```sql
SELECT
    eventTime, eventSource, userIdentity.type, userIdentity.principalid, userIdentity.accountid, userIdentity.username,
    element_at( serviceEventDetails, 'role_name' ) as role_name,
    element_at( serviceEventDetails, 'account_id' ) as account_id
FROM
    *****
WHERE
    eventTime > '2022-11-10 00:00:00'
    AND (eventSource = 'sso.amazonaws.com' AND element_at(serviceEventDetails, 'role_name' ) = 'AWSAdministratorAccess' )
ORDER BY eventTime DESC LIMIT 100
```
