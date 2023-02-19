from aws_cdk import aws_iam as iam, aws_s3 as _s3, core


class DatalakeWithCloudTrailStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create the IAM User to Be the Data Analyst
        iam.User(
            self,
            id="datalake_user",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonAthenaFullAccess")
            ],
        ).attach_inline_policy(
            iam.Policy(
                self,
                id="DatalakeUserBasic",
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "lakeformation:GetDataAccess",
                            "glue:GetTable",
                            "glue:GetTables",
                            "glue:SearchTables",
                            "glue:GetDatabase",
                            "glue:GetDatabases",
                            "glue:GetPartitions",
                        ],
                        effect=iam.Effect.ALLOW,
                        resources=["*"],
                    )
                ],
            )
        )

        # Create an Amazon S3 Bucket for the Data Lake
        _s3.Bucket(self, "cc-hiroga-datalake-cloudtrail")
