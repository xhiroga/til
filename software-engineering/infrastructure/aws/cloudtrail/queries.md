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

## Detect manually changing Security Group

```sql
SELECT
    eventName, userIdentity.arn AS user, sourceIPAddress, eventTime,
    element_at(requestParameters, 'groupID') AS securityGroup,
    element_at(requestParameters, 'ipPermissions') AS ipPermissions
FROM
    cdc27250-6e9b-4cfb-afaf-4c14eacd649d
WHERE
    (element_at(requestParameters, 'groupId') LIKE '%sg-%')
    AND sourceIPAddress != 'cloudformation.amazonaws.com'
    AND eventTime > '2022-12-20 00:00:00'
ORDER
    BY eventTime ASC
```

- `groupId`: キャメルケース
- `sourceIPAddress != 'cloudformation.amazonaws.com'`: CFnのケースを除外
    - ちなみに `!=` の代わりに `<>` も利用可能

なお、マネジメントコンソールからの操作の場合、`sourceIPAddress` が `AWS Internal` になる。

# Investigate manually created resources

```sql
FROM
    cdc27250-6e9b-4cfb-afaf-4c14eacd649d
WHERE
    (eventName LIKE '%Create%')
    AND resources IS NOT NULL
    AND userIdentity.sessioncontext.sessionissuer.username NOT LIKE 'AWSServiceRole%'
    AND userIdentity.sessioncontext.sessionissuer.username IS NOT NULL
    AND userIdentity.invokedBy != 'backup.amazonaws.com'
    AND eventName != 'CreateGrant'
    AND sourceIpAddress != 'cloudformation.amazonaws.com'
    AND eventTime > '2022-12-01 00:00:00'
ORDER
    BY eventTime ASC
```

- `userIdentity.invokedBy != 'backup.amazonaws.com'` または `eventName != 'CreateGrant'`: AWS Backup実行時、暗号化されたデータベースを複合するための権限を作成するため頻出。
