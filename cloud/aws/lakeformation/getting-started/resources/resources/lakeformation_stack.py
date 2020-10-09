from aws_cdk import aws_iam as iam, core


class LakeFormationStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create an IAM Role for Workflows
        lakeformation_workflow_role_id = "LakeFormationWorkflowRole"
        access = iam.PolicyDocument(
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
            inline_policies={"Access": access},
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSGlueServiceRole"
                )
            ],
        )

