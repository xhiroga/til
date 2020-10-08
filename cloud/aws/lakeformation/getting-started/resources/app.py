#!/usr/bin/env python3

from aws_cdk import core

from resources.lakeformation_stack import LakeFormationStack


app = core.App()
LakeFormationStack(app, "lakeformation-getting-stated")

app.synth()
