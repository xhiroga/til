from aws_cdk import aws_iam as iam, core


class LakeFormationStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create an IAM Role for Workflows
        lakeformation_workflow_role_id = "LakeFormationWorkflowRole"
        lakeFormationWorkflow = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "lakeformation:GetDataAccess",
                        "lakeformation:GrantPermissions",
                    ],
                    effect=iam.Effect.ALLOW,
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    effect=iam.Effect.ALLOW,
                    resources=[
                        f"arn:aws:iam::{self.account}:role/{lakeformation_workflow_role_id}"
                    ],
                ),
            ]
        )
        iam.Role(
            self,
            id=lakeformation_workflow_role_id,
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSGlueServiceRole"
                )
            ],
            inline_policies={"LakeFormationWorkflow": lakeFormationWorkflow},
        )

        # Create a Data Lake Administrator
        iam.User(
            self,
            id="DataLakeAdministrator",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSLakeFormationDataAdmin"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSGlueConsoleFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "CloudWatchLogsReadOnlyAccess"
                ),
            ],
        ).attach_inline_policy(
            iam.Policy(
                self,
                id="LakeFormationSLR",
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "iam:CreateServiceLinkedRole",
                        ],
                        effect=iam.Effect.ALLOW,
                        resources=["*"],
                        conditions={
                            "StringEquals": {
                                "iam:AWSServiceName": "lakeformation.amazonaws.com"
                            }
                        },
                    ),
                    iam.PolicyStatement(
                        actions=["iam:PutRolePolicy"],
                        effect=iam.Effect.ALLOW,
                        resources=[
                            f"arn:aws:iam::{self.account}:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
                        ],
                    ),
                ],
            )
        )
