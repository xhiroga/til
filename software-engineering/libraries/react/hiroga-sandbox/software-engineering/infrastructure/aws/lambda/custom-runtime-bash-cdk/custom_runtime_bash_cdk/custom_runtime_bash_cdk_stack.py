from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda
from aws_cdk import core


class CustomRuntimeBashCdkStack(core.Stack):

    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        service_role = iam.Role(
            self, "BashFunctionServiceRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")]
        )

        lambda_layer = aws_lambda.LayerVersion(
            self, "BashLayer",
            code=aws_lambda.Code.from_asset("./src/layer"),
            description="Bash runtime"
        )

        lambda_fn = aws_lambda.Function(
            self, "BashFunction",
            code=aws_lambda.Code.from_asset("./src/function"),
            layers=[lambda_layer],
            handler="function.handler",
            timeout=core.Duration.seconds(300),
            runtime=aws_lambda.Runtime.PROVIDED,
            memory_size=128,
            role=service_role
        )
