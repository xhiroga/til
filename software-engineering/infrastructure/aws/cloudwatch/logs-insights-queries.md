# CloudWatch Logs Insights

## Tasks failed container health check by EventBridge and CloudWatch Logs Insights

```
fields detail.stoppingAt as `StoppingAt(UTC)`, detail.startedAt as `StartedAt(UTC)`, detail.containers.0.name as Container0, detail.containers.1.name as Container1, detail.containers.0.networkInterfaces.0.privateIpv4Address as IpAddress, detail.lastStatus as LastStatus, detail.stoppedReason as StoppedReason
| filter detail.stoppedReason like /Task failed container health checks/
| sort @timestamp desc
```

- `detail.stoppingAt` and  `detail.startedAt` are in UTC.
- IPv4Addressは `detail.attachments` からも取得できるが、 `detail.attachments.1...` の場合と `detail.attachments.0...` の場合があって安定しない。
