import os
import shutil
import tempfile
from os import (chmod, chown,)
from os.path import isfile, isdir

from behave import given, then


@given('migration dir')
def step_impl(context):
    try:
        shutil.rmtree(context.migr_dir)
    except Exception:
        pass
    context.migr_dir = tempfile.mkdtemp()
    os.mkdir(os.path.join(context.migr_dir, 'migrations'))
    os.mkdir(os.path.join(context.migr_dir, 'callbacks'))


@given('migration dir has wrong access rights')
def step_impl(context):
    chmod(context.migr_dir, 0o700)


@then('migration dir exists')
def step_impl(context):
    file_path = os.path.join(
        context.migr_dir, 'migrations', 'V0123__test0123.sql')
    assert isfile(file_path)


@then('migration dir has migration file')
def step_impl(context):
    file_path = os.path.join(
        context.migr_dir, 'migrations')
    assert isdir(file_path)
