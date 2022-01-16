#!/usr/bin/env python3

from aws_cdk import core

from resources.setting_up_stack import SettingUpStack
from resources.datalake_with_cloudtrail_stack import DatalakeWithCloudTrailStack

app = core.App()
SettingUpStack(app, "lakeformation-getting-stated")
DatalakeWithCloudTrailStack(app, "cloudtrail-tutorial")

app.synth()
