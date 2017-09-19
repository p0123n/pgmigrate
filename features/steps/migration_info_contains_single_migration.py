from behave import then
from pgmigrate import _get_info


@then("migration info contains single migration")
def step_impl(context):
    class Config:
        cursor = context.conn.cursor()
        baseline = 0
        base_dir = context.migr_dir
        target = 1

    info = _get_info(Config)
    assert list(info.values())[0]['version'] == 1
    assert list(info.values())[0]['description'] == 'Single migration'
