from aws_cdk import (
    aws_iam as iam,
    aws_s3 as _s3,
    core
)


class LakeFormationStack(core.Stack):

    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # IAM Role for Workflows
        iam.Role(self, id='LakeFormationWorkflowRole', assumed_by=iam.ServicePrincipal('glue.amazonaws.com'))
