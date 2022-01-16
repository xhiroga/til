#!/usr/bin/env python3

from aws_cdk import core

from custom_runtime_bash_cdk.custom_runtime_bash_cdk_stack import CustomRuntimeBashCdkStack

app = core.App()
CustomRuntimeBashCdkStack(app, "custom-runtime-bash-cdk", env=core.Environment(region="ap-northeast-1"))

app.synth()
